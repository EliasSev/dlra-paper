"""
Exact direct solve of the full-rank Tikhonov problem -- the reference solution
the iterative solvers (CG, DLRA-*) are compared against when an "exact" answer,
not just a converged one, is wanted.

At the minimum of Phi the gradient vanishes, giving the normal equations

    H x = b,   H = K^T M_ds K + lambda^2 W^T M_dx W,   b = K^T M_ds y,

with W = diag(w). H is symmetric positive definite, so it is factored once with a
dense Cholesky and back-substituted. The data term K^T M_ds K is assembled through
the SVD factors K = U S V^T (cost O(N^2 r), never forming a dense K^T M K product),
and the regularization term is sparse. The factorization is cached, so solving for
several right-hand sides y at fixed (w, lambda_) pays for Cholesky only once.

This is a full-rank solver: dense and O(N^3), intended for the small meshes on
which the low-rank solvers are validated -- not a scalable method.
"""
import numpy as np

from typing import Optional
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.linalg import cho_factor, cho_solve

from dlra.solvers.base import TikhonovSolver


class DirectSolver(TikhonovSolver):
    """Solve the full-rank Tikhonov problem exactly via Cholesky of the normal equations."""

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            verbose: bool = False,
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} exactly for the given lambda_.

        y, NDArray     : The observed data (1D array).
        w, NDArray     : Tikhonov regularization weights (1D array).
        lambda_, float : Tikhonov regularization parameter (pass lambda, it is
                         squared internally).
        verbose, bool  : Print the relative residual ||H x - b|| / ||b||.

        returns: Solution vector x = vec(X) (1D array), the exact Tikhonov minimizer.
        """
        lambda_ = lambda_**2

        cho = self._factorize(w, lambda_)
        b = self.V @ (self.S * (self.UT @ (self.M_ds @ y)))
        x = cho_solve(cho, b)

        self.niter = 1
        if verbose:
            H_x = self._apply_H_vec(x, w, lambda_)
            rel_res = np.linalg.norm(H_x - b) / np.linalg.norm(b)
            print(f"Direct solve [rel. residual ||Hx - b||/||b|| = {rel_res:.3e}]")
        return x

    # --- internals ----------------------------------------------------------

    def _factorize(self, w: NDArray, lambda_: float):
        """
        Cholesky factor of H = K^T M_ds K + lambda_ * W^T M_dx W, cached by
        (w, lambda_) so repeated right-hand sides reuse it.
        """
        key = (lambda_, w.tobytes())
        if getattr(self, "_cache_key", None) != key:
            self._cache = cho_factor(self._build_H(w, lambda_))
            self._cache_key = key
        return self._cache

    def _build_H(self, w: NDArray, lambda_: float) -> NDArray:
        """Assemble the dense (N, N) Hessian H = K^T M_ds K + lambda_ * W^T M_dx W."""
        # Data term K^T M_ds K = V C V^T with C = S (U^T M_ds U) S  (r x r).
        C = (self.S[:, None] * (self.UT @ (self.M_ds @ self.U))) * self.S[None, :]
        H = self.V @ C @ self.VT

        # Regularization term lambda_ * diag(w) M_dx diag(w), sparse.
        Dw = diags(w)
        H += lambda_ * (Dw @ self.M_dx @ Dw).toarray()
        return H

    def _apply_H_vec(self, x: NDArray, w: NDArray, lambda_: float) -> NDArray:
        """H x in flat FE coordinates (residual check; avoids the vec<->matrix round trip)."""
        Kx = self.U @ (self.S * (self.VT @ x))
        H_data = self.V @ (self.S * (self.UT @ (self.M_ds @ Kx)))
        H_reg = lambda_ * (w * (self.M_dx @ (w * x)))
        return H_data + H_reg
