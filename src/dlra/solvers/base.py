"""
Shared machinery for the CG / low-rank solver family: Frobenius helpers and the
`TikhonovSolver` base class (factor storage, vec <-> matrix indexing, the objective's
gradient and Hessian applied through the SVD factors of K, tangent-space
projection, and rank truncation).

The solvers consume K in factored form K = U S V^T (scope: K is *given*; see
CLAUDE.md). By default the factors are computed from the exact dense K of a
`ForwardOperator` via a truncated SVD; alternatively, precomputed factors
(e.g. from `dlra.rsvd`) can be passed directly.
"""
import numpy as np

from numpy.typing import NDArray
from typing import Optional, Union
from abc import ABC, abstractmethod
from scipy.sparse.linalg import LinearOperator

from dlra.io import progress_bar
from dlra.problem.fem import ForwardOperator


def frobenius2(A):
    """Compute the squared Frobenius norm of A."""
    a = A.ravel()
    return np.dot(a, a)


def inner_F(A, B):
    """Compute the Frobenius inner product between A and B."""
    return np.dot(A.ravel(), B.ravel())


class TikhonovSolver(ABC):
    """
    Base class for solvers of the Tikhonov-regularized inverse problem

        min_x  1/2 |Kx - y|²_{M_ds} + λ²/2 |Wx|²_{M_dx},

    using the SVD K = U S V^T of the forward operator K. The class implements: `gradient`, `apply_H` (Hessian), and `initial_X`, and other helper methods the subclasses.
    """

    def __init__(
            self,
            operator: ForwardOperator,
            k: Optional[int] = None,
            factors: Optional[tuple[NDArray, NDArray, NDArray]] = None,
            x_true: Optional[NDArray] = None,
            svd_tol: float = 1e-12,
        ) -> None:
        """
        operator, ForwardOperator : The forward problem (K, M_dx, M_ds).
        k, int | None             : Target rank of the K factors. None keeps the
                                    numerical rank (relative tol `svd_tol`).
        factors, (U, S, Vt)|None  : Precomputed SVD factors of K (S as a 1D
                                    array). Overrides the SVD of operator.K.
        x_true, NDArray | None    : Ground truth, enables error tracking.
        """
        self.Vh = operator.Vh
        self.M_dx, self.M_ds = operator.M_dx, operator.M_ds

        if factors is not None:
            U, S, Vt = factors
        else:
            U, S, Vt = np.linalg.svd(operator.K, full_matrices=False)
            r = int(np.sum(S > S[0] * svd_tol)) if k is None else k
            U, S, Vt = U[:, :r], S[:r], Vt[:r, :]

        self.U, self.S, self.VT = U, S, Vt
        self.UT, self.V = self.U.T, self.VT.T

        self._x_true = None
        self._X_true = None

        self.error = []     # Track the error
        self.residual = []  # Track residuals
        self.niter = 0      # Number of iterations to converge

        # Set up vec to matrix and matrix to vec utils
        coords = self.Vh.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(self.Vh.dim()))

        # Must be done after grid_indices is set up
        if x_true is not None:
            self.x_true = x_true

    @property
    def x_true(self) -> NDArray:
        if self._x_true is None:
            raise ValueError("'x_true' is not set!")
        return self._x_true

    @property
    def X_true(self) -> NDArray:
        if self._X_true is None:
            raise ValueError("'X_true' is not set!")
        return self._X_true

    @x_true.setter
    def x_true(self, value: NDArray) -> None:
        self._x_true = value
        self._X_true = self.vec_to_matrix(value)

    @X_true.setter
    def X_true(self, value: NDArray) -> None:
        self._x_true = self.matrix_to_vec(value)
        self._X_true = value

    def matrix_to_vec(self, X: NDArray) -> NDArray:
        return X.flatten()[self.dof_indices]

    def vec_to_matrix(self, x: NDArray) -> NDArray:
        return x[self.grid_indices].reshape((self.n, self.n))

    @abstractmethod
    def solve(self):
        """Solver method to be implemented by subclasses."""
        pass

    def _err0(self, X: NDArray) -> Optional[float]:
        """Initial error norm, or None when no ground truth is set."""
        if self._X_true is None:
            return None
        return np.sqrt(frobenius2(X - self.X_true))

    def _track_and_check(self, G, X, res0, err0, rtol, etol, i, max_iter, verbose):
        """Record residual/error, check convergence, print progress. Returns True if done."""
        rel_res = np.sqrt(frobenius2(G)) / res0
        self.residual.append(rel_res)

        if err0 is not None:
            rel_err = np.sqrt(frobenius2(X - self.X_true)) / err0
            self.error.append(rel_err)

        if rel_res < rtol:
            if verbose: print(f"Converged at iter {i} [rtol criteria: rel_res={rel_res:.3}]")
            return True

        if etol is not None and err0 is not None:
            if rel_err < etol:
                if verbose: print(f"Converged at iter {i} [etol criteria: rel_err={rel_err:.3}]")
                return True

        if verbose and ((i % 100 == 0) or (i == max_iter)):
            progress_bar(i, max_iter)
        return False

    def initial_X(
            self, seed: Optional[int], max_rank: int, X0: Union[str, NDArray]
        ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Generate an initial matrix X and its SVD of rank `max_rank`.
        """
        rng = np.random.default_rng(seed)

        # Custom 'X0' passed in by user
        if isinstance(X0, np.ndarray):
            Ux, sx, VxT = np.linalg.svd(X0, full_matrices=False)
            Sx = np.diag(sx)
            Vx = VxT.T

        elif isinstance(X0, str):
            if X0 == 'svd':
                X = rng.random((self.n, self.n)) * 1e-3
                Ux, sx, VxT = np.linalg.svd(X, full_matrices=False)
                Sx = np.diag(sx)
                Vx = VxT.T

            elif X0 == 'qr':
                Ux = rng.random((self.n, self.n))
                Vx = rng.random((self.n, self.n))
                Ux, _ = np.linalg.qr(Ux)
                Vx, _ = np.linalg.qr(Vx)

                # Mimic the singular values of X ~ Uniform(0, 1):
                # sigma_1 = 0.5 * n, sigma_2, ..., sigma_n = O(sqrt(n))
                sx = np.sqrt(self.n) * rng.random(self.n)
                sx[0] = 0.5 * self.n
                sx = np.sort(sx)[::-1] * 1e-3
                Sx = np.diag(sx)

            elif X0 == 'low-rank-qr':
                Ux = rng.random((self.n, self.n))
                Vx = rng.random((self.n, self.n))
                Ux, _ = np.linalg.qr(Ux)
                Vx, _ = np.linalg.qr(Vx)

                sx = np.sqrt(self.n) * rng.random(self.n)
                sx[0] = 0.5 * self.n
                sx = np.sort(sx)[::-1] * 1e-3
                sx[max_rank:] = 0
                Sx = np.diag(sx)

            elif X0 == 'householder':
                Ux = self.fast_orthogonal(rng, max_rank)
                Vx = self.fast_orthogonal(rng, max_rank)

                sx = np.sqrt(self.n) * rng.random(self.n)
                sx[0] = 0.5 * self.n
                Sx = np.diag(np.sort(sx)[::-1] * 1e-3)

            else:
                raise ValueError(f"Invalid 'X0': '{X0}'")

        else:
            raise ValueError(f"Invalid 'X0' type {type(X0)}")

        X = Ux @ Sx @ Vx.T
        return X, Ux, Sx, Vx

    def fast_orthogonal(self, rng, k):
        """
        Apply Householder transformations form an orthogonal Q.
        """
        Q = np.eye(self.n)
        for _ in range(k):
            v = rng.standard_normal(self.n)
            v /= np.linalg.norm(v)
            Q -= 2 * np.outer(v, v)
        return Q

    def gradient(self, X: NDArray, y: NDArray, w: NDArray, lambda_: float) -> NDArray:
        """
        Given the SVD K = U S V^T, compute the gradient of the cost
        function Phi with respect to X.
        """
        x = self.matrix_to_vec(X)
        r = self.U @ (self.S * (self.VT @ x)) - y

        grad_data = self.V @ (self.S * (self.UT @ (self.M_ds @ r)))
        grad_reg = lambda_ * (w * (self.M_dx @ (w * x)))

        return self.vec_to_matrix(grad_data + grad_reg)

    def apply_H(self, P: NDArray, w: NDArray, lambda_: float) -> NDArray:
        """
        Computes HP = mat[H vec(P)], where H is the Hessian of the cost function Phi.
        Here, H = K^T M_ds K + lambda * W^T M W.
        """
        p = self.matrix_to_vec(P)
        Kp = self.U @ (self.S * (self.VT @ p))

        # Compute (K^T M_ds K)p and (lambda * W^T M W)p
        H_data = self.V @ (self.S * (self.UT @ (self.M_ds @ Kp)))
        H_reg = lambda_ * (w * (self.M_dx @ (w * p)))
        return self.vec_to_matrix(H_data + H_reg)

    def truncate(
            self, U1: NDArray, S: NDArray, V1: NDArray, tol: float, max_rank: int = 1
        ) -> NDArray:
        """
        Truncates according to tolerance
        U1: (m x k) left factor
        S:  (k x k) matrix to re-SVD (can be diagonal or full)
        V1: (n x k) right factor
        tol: scalar tolerance (relative factor multiplied by norm(S))
        Returns: (U1_trunc, S_trunc, V1_trunc)
        """
        U_s, s_vals, Vh = np.linalg.svd(S, full_matrices=False)
        tol = tol * np.linalg.norm(S)

        # cumulative tail-sum test
        rmax = s_vals.size
        retained = rmax  # default keep all
        for j in range(rmax):
            tail_sum = np.sum(s_vals[j:rmax])
            if abs(tail_sum) < tol:
                retained = j
                break

        if max_rank is not None:
            retained = min(retained, int(max_rank))

        # Truncation / rotate factors
        U1 = U1 @ U_s
        V1 = V1 @ Vh.T

        S_trunc = np.diag(s_vals[:retained])
        U1_trunc = U1[:, :retained]
        V1_trunc = V1[:, :retained]

        return U1_trunc, S_trunc, V1_trunc

    def get_preconditioner(
            self, w: NDArray, lambda_: float, preconditioner: str
        ) -> Union[NDArray, LinearOperator]:
        """
        Build a preconditioner for the Hessian H (see `dlra.solvers.preconditioners`).
        Imported lazily so the heavy deps (pymatting, sksparse) are only required
        when a preconditioned solver is actually used.
        """
        from dlra.solvers.preconditioners import get_preconditioner
        return get_preconditioner(
            preconditioner,
            U=self.U, S=self.S, V=self.V,
            M_dx=self.M_dx, M_ds=self.M_ds,
            w=w, lambda_=lambda_,
        )

    def apply_P_inv(self, A: NDArray, P_inv: Union[NDArray, LinearOperator]) -> NDArray:
        """Compute mat[ P^{-1} vec(A) ]."""
        a = self.matrix_to_vec(A)
        if isinstance(P_inv, LinearOperator):
            return self.vec_to_matrix(P_inv @ a)
        else:
            return self.vec_to_matrix(P_inv * a)
