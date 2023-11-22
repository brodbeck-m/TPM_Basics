# --- Imports ---
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from typing import List

import dolfinx
import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
import dolfinx.mesh as dmesh
import dolfinx.nls.petsc as dnls_petsc
from dolfinx.mesh import CellType, DiagonalType, create_rectangle
import ufl

from petsc4py.PETSc import ScalarType

# --- Input parameters ---
# Geometry

# Material
mat_EhS = 1.0e7
mat_nuhS = 0.25
mat_ktD = 1.0e-6

# Discretisation
sdisc_nelmt = [9, 72]
sdisc_eorder = [1, 1]
tdisc_dt = 0.005

# Load
bc_qtop = -10000


# --- Auxiliaries ---
def create_geometry_rectangle(
    l_domain: List[float],
    n_elmt: List[int],
    diagonal: DiagonalType = DiagonalType.left,
):
    # --- Create mesh
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([l_domain[0], l_domain[1]])],
        [n_elmt[0], n_elmt[1]],
        cell_type=CellType.triangle,
        diagonal=diagonal,
    )
    tol = 1.0e-14
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], l_domain[0])),
        (4, lambda x: np.isclose(x[1], l_domain[1])),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dmesh.locate_entities(mesh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dmesh.meshtags(
        mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    return mesh, facet_tag, ds


def set_boundary_conditions(facet_tags, V, Vu, Vp):
    # Interpolate dirichlet conditions
    def bc_vec_zero(x):
        return np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    def bc_scalar_zero(x):
        return np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

    uD = dfem.Function(Vu)
    uD.interpolate(bc_vec_zero)
    pD = dfem.Function(Vp)
    pD.interpolate(bc_scalar_zero)

    list_bc = []

    # Displacement boundaries
    # Bottom
    facets = facet_tags.indices[facet_tags.values == 2]
    dofs = dfem.locate_dofs_topological((V.sub(0).sub(1), Vu.sub(1)), 1, facets)
    list_bc.append(dfem.dirichletbc(uD, dofs, V.sub(0)))

    # Left
    facets = facet_tags.indices[facet_tags.values == 1]
    dofs = dfem.locate_dofs_topological((V.sub(0).sub(0), Vu.sub(0)), 1, facets)
    list_bc.append(dfem.dirichletbc(uD, dofs, V.sub(0)))

    # Right
    facets = facet_tags.indices[facet_tags.values == 3]
    dofs = dfem.locate_dofs_topological((V.sub(0).sub(0), Vu.sub(0)), 1, facets)
    list_bc.append(dfem.dirichletbc(uD, dofs, V.sub(0)))

    # Pressure boundaries
    # Top
    facets = facet_tags.indices[facet_tags.values == 4]
    dofs = dfem.locate_dofs_topological((V.sub(1), Vp), 1, facets)
    list_bc.append(dfem.dirichletbc(pD, dofs, V.sub(1)))

    return list_bc


# --- Problem setup ---
# --- Set geometry
domain, facet_tags, ds = create_geometry_rectangle([1.0, 5.0], [9, 70])

# --- Set function-space
Pu = ufl.VectorElement("CG", domain.ufl_cell(), sdisc_eorder[0])
Pp = ufl.FiniteElement("CG", domain.ufl_cell(), sdisc_eorder[1])

V_up = dfem.FunctionSpace(domain, ufl.MixedElement(Pu, Pp))

V_u, up_to_u = V_up.sub(0).collapse()
Vu = dfem.FunctionSpace(domain, Pu)
Vp = dfem.FunctionSpace(domain, Pp)

uh = dfem.Function(V_up)
uh_n = dfem.Function(V_u)

# --- Set weak form
# Trial- and test functions
u, p = ufl.split(uh)
v_u, v_p = ufl.TestFunctions(V_up)

# Linearized Green-Lagrange strain
EtS = ufl.sym(ufl.grad(u))

# Solid velocity
vtS = (u - uh_n) / tdisc_dt

# Stress
mat_lhs = (mat_nuhS * mat_EhS) / ((1 + mat_nuhS) * (1 - 2 * mat_nuhS))
mat_mhs = mat_EhS / (2 * (1 + mat_nuhS))

PhSE = 2.0 * mat_mhs * EtS + mat_lhs * ufl.div(u) * ufl.Identity(len(u))
P = PhSE - p * ufl.Identity(len(u))

# Fluid flux
nhFwtFS0S = mat_ktD * ufl.grad(p)

# Load
qtop = dfem.Constant(domain, ScalarType(bc_qtop))
load = qtop * ufl.FacetNormal(domain)

# Residual
res_BLM = ufl.inner(P, ufl.sym(ufl.grad(v_u))) * ufl.dx
res_BMO = (ufl.div(vtS) * v_p + ufl.inner(nhFwtFS0S, ufl.grad(v_p))) * ufl.dx
load_term = ufl.inner(v_u, load) * ds(4)

# Add volumetric contributions of weak form
weak_form = res_BLM + res_BMO - load_term

# --- Set boundary conditions
list_bc = set_boundary_conditions(facet_tags, V_up, Vu, Vp)

# --- Set Solver
# Initialise non-linear problem
nl_problem = dfem_petsc.NonlinearProblem(weak_form, uh, list_bc)

# Initialise newton solver
solver = dnls_petsc.NewtonSolver(domain.comm, nl_problem)
solver.atol = 1e-10
solver.rtol = 1e-10
solver.convergence_criterion = "incremental"

# Configure mumps
solver.krylov_solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.krylov_solver.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")

# Initialize physical time
time = 0.0

# Initialize history values
uh_n.x.array[:] = 0.0

# Initialize export ParaView
outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test_terzaghi-NewtonSolver.xdmf", "w")
outfile.write_mesh(domain)

# --- Solve problem ---
duration_solve = 0.0
for n in range(300):
    # Update time
    time = time + tdisc_dt

    # Solve newton-steps
    duration_solve -= MPI.Wtime()
    niter, converged = solver.solve(uh)
    duration_solve += MPI.Wtime()

    PETSc.Sys.Print(
        "Phys. Time {:.4f}, Calc. Time {:.4f}, Num. Iter. {}".format(
            time, duration_solve, niter
        )
    )

    uh_n.x.array[:] = uh.x.array[up_to_u]

    u = uh.sub(0).collapse()
    p = uh.sub(1).collapse()

    u.name = "u_h"
    outfile.write_function(u, time)
    p.name = "p_h"
    outfile.write_function(p, time)

outfile.close()
