# --- Imports ---
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from typing import List

import basix
import basix.ufl

from dolfinx import fem, io, mesh, default_real_type
import dolfinx.fem.petsc as fem_petsc
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
    diagonal: mesh.DiagonalType = mesh.DiagonalType.left,
):
    # --- Create mesh
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([l_domain[0], l_domain[1]])],
        [n_elmt[0], n_elmt[1]],
        cell_type=mesh.CellType.triangle,
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
        facets = mesh.locate_entities(msh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        msh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    return msh, facet_tag, ds


def set_boundary_conditions(facet_tags, V, Vu, Vp):
    # The spatial dimension
    gdim = V.mesh.geometry.dim

    # Required connectivity
    V.mesh.topology.create_connectivity(gdim - 1, gdim)

    # Interpolate dirichlet conditions
    def bc_vec_zero(x):
        return np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    def bc_scalar_zero(x):
        return np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

    uD = fem.Function(Vu)
    uD.interpolate(bc_vec_zero)
    pD = fem.Function(Vp)
    pD.interpolate(bc_scalar_zero)

    list_bc = []

    # Displacement boundaries
    # Bottom
    facets = facet_tags.indices[facet_tags.values == 2]
    dofs = fem.locate_dofs_topological((V.sub(0).sub(1), Vu.sub(1)), 1, facets)
    list_bc.append(fem.dirichletbc(uD, dofs, V.sub(0)))

    # Left
    facets = facet_tags.indices[facet_tags.values == 1]
    dofs = fem.locate_dofs_topological((V.sub(0).sub(0), Vu.sub(0)), 1, facets)
    list_bc.append(fem.dirichletbc(uD, dofs, V.sub(0)))

    # Right
    facets = facet_tags.indices[facet_tags.values == 3]
    dofs = fem.locate_dofs_topological((V.sub(0).sub(0), Vu.sub(0)), 1, facets)
    list_bc.append(fem.dirichletbc(uD, dofs, V.sub(0)))

    # Pressure boundaries
    # Top
    facets = facet_tags.indices[facet_tags.values == 4]
    dofs = fem.locate_dofs_topological((V.sub(1), Vp), 1, facets)
    list_bc.append(fem.dirichletbc(pD, dofs, V.sub(1)))

    return list_bc


# --- Problem setup ---
# --- Set geometry
domain, facet_tags, ds = create_geometry_rectangle([1.0, 5.0], [9, 70])

# --- Set function-space
Pu = basix.ufl.element(
    "Lagrange",
    domain.basix_cell(),
    degree=sdisc_eorder[0],
    shape=(domain.geometry.dim,),
    dtype=default_real_type,
)
Pp = basix.ufl.element(
    "Lagrange", domain.basix_cell(), degree=sdisc_eorder[1], dtype=default_real_type
)

V_up = fem.functionspace(domain, basix.ufl.mixed_element([Pu, Pp]))

V_u, up_to_u = V_up.sub(0).collapse()
V_p, up_to_p = V_up.sub(1).collapse()

uh, uh_u, uh_p = fem.Function(V_up), fem.Function(V_u), fem.Function(V_p)

# --- Set weak form
# Trial- and test functions
u, p = ufl.TrialFunctions(V_up)
v_u, v_p = ufl.TestFunctions(V_up)

# Linearized Green-Lagrange strain
EtS = ufl.sym(ufl.grad(u))

# Solid velocity
vtS = (u - uh_u) / tdisc_dt

# Stress
mat_lhs = (mat_nuhS * mat_EhS) / ((1 + mat_nuhS) * (1 - 2 * mat_nuhS))
mat_mhs = mat_EhS / (2 * (1 + mat_nuhS))

PhSE = 2.0 * mat_mhs * EtS + mat_lhs * ufl.div(u) * ufl.Identity(len(u))
P = PhSE - p * ufl.Identity(len(u))

# Fluid flux
nhFwtFS0S = mat_ktD * ufl.grad(p)

# Load
qtop = fem.Constant(domain, ScalarType(bc_qtop))
load = qtop * ufl.FacetNormal(domain)

# Residual
res_BLM = ufl.inner(P, ufl.sym(ufl.grad(v_u))) * ufl.dx
res_BMO = (ufl.div(vtS) * v_p + ufl.inner(nhFwtFS0S, ufl.grad(v_p))) * ufl.dx
load_term = ufl.inner(v_u, load) * ds(4)

# Add volumetric contributions of weak form
weak_form = res_BLM + res_BMO - load_term

# --- Set boundary conditions
list_bc = set_boundary_conditions(facet_tags, V_up, V_u, V_p)

# --- Set Solver
# Compile forms
a = fem.form(ufl.lhs(weak_form))
l = fem.form(ufl.rhs(weak_form))

# Assembly equation system
A = fem_petsc.assemble_matrix(a, bcs=list_bc)
A.assemble()

# Initialise RHS
L = fem_petsc.create_vector(fem.extract_function_spaces(l))

# Set solver
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setTolerances(rtol=1e-10, atol=1e-10)

# Configure mumps
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")

# --- Solve problem ---
# Initialize physical time
time = 0.0

# Initialize history values
uh_u.x.array[:] = 0.0

# Initialize export ParaView
uh_u.name = "u_h"
outfile_u = io.VTXWriter(MPI.COMM_WORLD, "terzaghi-u.bp", [uh_u], engine="BP4")

uh_p.name = "p_h"
outfile_p = io.VTXWriter(MPI.COMM_WORLD, "terzaghi-p.bp", [uh_p], engine="BP4")

# Time loop
duration_solve = 0.0
for n in range(300):
    # Update time
    time = time + tdisc_dt

    # Calculate current solution
    duration_solve -= MPI.Wtime()

    # Assemble RHS
    with L.localForm() as loc_L:
        loc_L.set(0)

    fem_petsc.assemble_vector(L, l)
    fem.apply_lifting(L, [a], [list_bc])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    fem.set_bc(L, list_bc)

    # Solve equation system
    solver(L, uh.x.petsc_vec)
    uh.x.scatter_forward()

    duration_solve += MPI.Wtime()

    PETSc.Sys.Print("Phys. Time {:.4f}, Calc. Time {:.4f}".format(time, duration_solve))

    uh_u.x.array[:] = uh.x.array[up_to_u]
    uh_p.x.array[:] = uh.x.array[up_to_p]

    outfile_u.write(time)
    outfile_p.write(time)

outfile_u.close()
outfile_p.close()
