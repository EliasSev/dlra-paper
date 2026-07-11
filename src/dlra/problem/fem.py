"""
Finite-element assembly and the exact forward operator for the elliptic problem.

Assembles the FE matrices of the forward problem from the PDE
    -div(sigma grad u) + c u = f
with homogeneous Neumann BCs (boundary DOFs, stiffness matrix A, volume mass
matrix M_dx, boundary mass matrix M_ds, trace operator T), and exposes the exact
forward operator K = T A^{-1} M and its adjoint on top of them.

Cheap sparse matrices are assembled eagerly in `__init__`. The expensive pieces
(the Cholesky factorization of A, the dense K / K*, the SVD-based weights) are
lazy `cached_property`s: they are built the first time they are accessed and
cached thereafter, so an instance created only to grab M_dx never pays for them.
"""
import numpy as np
from typing import Optional
from functools import cached_property
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, csc_matrix
from sksparse.cholmod import cholesky
from fenics import (
    FunctionSpace, DirichletBC, Constant, TrialFunction, TestFunction,
    dot, grad, dx, ds, assemble, as_backend_type, set_log_level
)
set_log_level(30)


class ForwardOperator:
    """
    FE assembly plus the exact forward operator K = T A^{-1} M.

    Vh:         Scalar CG1 function space.
    sigma, c:   Coefficients of -div(sigma grad u) + c u = f.

    Eager attributes: bdofs, N, N_b, A, M_dx, M_ds.
    Lazy properties:  T, chol_A, K, K_star, weights.
    """

    def __init__(self, Vh: FunctionSpace, sigma: float = 1.0, c: float = 1.0):
        self.Vh = Vh
        self.sigma = sigma
        self.c = c

        self.bdofs = self._assemble_boundary_dofs()
        self.N = Vh.dim()
        self.N_b = len(self.bdofs)

        self.A = self._assemble_A()
        self.M_dx = self._assemble_M_dx()
        self.M_ds = self._assemble_M_ds()

        self._T: Optional[csr_matrix] = None

    # --- assembly -----------------------------------------------------------

    def _assemble_boundary_dofs(self) -> NDArray:
        """Boundary DOF indices, sorted ascending."""
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.Vh, Constant(0.0), boundary)
        return np.array(sorted(bc.get_boundary_values().keys()), dtype=int)

    def _assemble_sparse(self, form) -> csr_matrix:
        """Assemble a bilinear form and return it as a scipy CSR matrix."""
        mat = as_backend_type(assemble(form)).mat()
        indptr, indices, data = mat.getValuesCSR()
        return csr_matrix((data, indices, indptr), shape=(self.N, self.N))

    def _assemble_A(self) -> csr_matrix:
        """Stiffness matrix: a(u, v) = sigma <grad u, grad v> + c <u, v>."""
        u, v = TrialFunction(self.Vh), TestFunction(self.Vh)
        a = (Constant(self.sigma) * dot(grad(u), grad(v)) + Constant(self.c) * u * v) * dx
        return self._assemble_sparse(a)

    def _assemble_M_dx(self) -> csr_matrix:
        """Volume mass matrix: m(u, v) = <u, v>_dx."""
        u, v = TrialFunction(self.Vh), TestFunction(self.Vh)
        return self._assemble_sparse(u * v * dx)

    def _assemble_M_ds(self) -> csr_matrix:
        """Boundary mass matrix, restricted to boundary DOFs: (N_b, N_b)."""
        u, v = TrialFunction(self.Vh), TestFunction(self.Vh)
        M_full = self._assemble_sparse(u * v * ds)
        return M_full[self.bdofs, :][:, self.bdofs]

    @property
    def T(self) -> csr_matrix:
        """Trace operator T (N_b, N): picks out the boundary DOFs of a vector."""
        if self._T is None:
            rows = np.arange(self.N_b)
            self._T = csr_matrix(
                (np.ones(self.N_b), (rows, self.bdofs)), shape=(self.N_b, self.N)
            )
        return self._T

    # --- forward operator ---------------------------------------------------

    @cached_property
    def chol_A(self):
        """Sparse Cholesky factorization of A."""
        return cholesky(self.A.tocsc())

    def apply_K(self, x: NDArray) -> NDArray:
        """
        Apply the forward operator: y = K x = T A^{-1} M x.
        """
        u = self.chol_A.solve_A(self.M_dx @ x)
        return u[self.bdofs]

    @cached_property
    def K(self) -> NDArray:
        """
        Dense forward operator K = T A^{-1} M, shape (N_b, N).

        Built as (M A^{-1} T^T)^T via N_b solves against the sparse RHS T^T.
        """
        T_T = csc_matrix(
            (np.ones(self.N_b), (self.bdofs, np.arange(self.N_b))),
            shape=(self.N, self.N_b),
        )
        Z = self.chol_A.solve_A(T_T.toarray())   # (N, N_b) dense
        return np.asarray((self.M_dx @ Z).T)     # (N_b, N) dense

    @cached_property
    def K_star(self) -> NDArray:
        """
        Dense adjoint operator K* = A^{-1} T^T M_ds, shape (N, N_b).

        N_b solves of A against the sparse RHS T^T M_ds.
        """
        T_T = csc_matrix(
            (np.ones(self.N_b), (self.bdofs, np.arange(self.N_b))),
            shape=(self.N, self.N_b),
        )
        rhs = (T_T @ self.M_ds).toarray()        # (N, N_b) dense
        return np.asarray(self.chol_A.solve_A(rhs))

    @cached_property
    def weights(self) -> NDArray:
        """
        Elvetun-Nielsen regularization weights as a 1D array w = diag(W),
            w_i^2 = (Vr Vr^T M_dx Vr Vr^T)_ii / M_ii.
        """
        return self._weights()

    def _weights(self, tol: float = 1e-12) -> NDArray:
        _, s, Vt = np.linalg.svd(self.K, full_matrices=False)
        r = np.sum(s > (s[0] * tol))     # numerical rank of K
        Vr = Vt[:r, :].T

        C = Vr.T @ self.M_dx @ Vr        # (r, r)
        w_sq = np.sum(Vr * (Vr @ C), axis=1)

        volumes = np.array(self.M_dx.sum(axis=1)).flatten()
        return np.sqrt(np.maximum(w_sq, 0)) / volumes
