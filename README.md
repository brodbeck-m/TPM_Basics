# TPM_Basics
A basic solver for poroelasticity, using a non-linear variational problem. The governing equations are currently based on Biots theory bun can be directly extended to non-linear cases.

## Instalation
For running the code, the official DOLFINx container is required at [dockerhub][1]. Pull the latest (stable) image by 
```bash
docker pull dolfinx/dolfinx:stable
```
After downloading the conatiner run VSCode within the folder. It will then set-up a dev environement. All these steps require a working docker instalation.

[1]: https://hub.docker.com/u/dolfinx "dockerhub"
