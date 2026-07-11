"""
Riemannian steepest descent on the fixed-rank manifold for the generalized
Sylvester-like equation

    L[X] := A1 X + X A2 + A3 X A4 = Y,

following `notes_low_rank_SD.tex` ("DLRT with exact line search"). L is assumed
self-adjoint with SPD Hessian; the rank-r solution is parametrized as
X = U S V^T and every quantity is handled through its low-rank factors -- the
full n x n matrices X, G[X], L[D] are never formed. Per step:

    1. residual R_k = -P(X_k) G[X_k]     (Riemannian gradient, tangent factors)
    2. converged when ||R_k||_F <= tol
    3. direction D_k = R_k
    4. projected Hessian P(X_k) L[D_k]   (tangent factors)
    5. exact line search  alpha_k = ||R_k||_F^2 / <D_k, P(X_k) L[D_k]>
    6. DLRT (basis-update & Galerkin) step + retraction via a small r x r SVD

Two classes:

- `LowRankSylvesterSD`  : the note verbatim, for an explicit Sylvester-structured
  L given by A1..A4. Standalone and fully factored (numpy only; A_i may be
  scipy sparse).
- `DynamicalLowRankSD`  : the same algorithm as a `TikhonovSolver` for the paper's
  inverse problem, where L is the Hessian H = K^T M_ds K + lambda^2 W^T M W and
  Y = mat(K^T M_ds y). H has no Sylvester structure, so G and H[D] are applied
  through `gradient`/`apply_H` on full matrices (like the other solvers); the
  Riemannian-SD/DLRT iteration itself follows the note.
"""
import numpy as np

from typing import Optional, Union
from numpy.typing import NDArray

from dlra.io import progress_bar
from dlra.solvers.base import TikhonovSolver, frobenius2, inner_F


class LowRankSylvesterSD:
    """
    Fixed-rank Riemannian steepest descent for A1 X + X A2 + A3 X A4 = Y.

    A1, A2, A3, A4 : (n, n) arrays / scipy sparse, or None for a missing term
                     (A3/A4 must be both given or both None). The induced
                     operator L must be self-adjoint (A_i symmetric suffices).
    Y              : Right-hand side, either a dense (n, n) array or a factored
                     pair (Yl, Yr) with Y = Yl @ Yr.T.
    X_true         : Optional dense ground truth; enables error tracking.
    """

    def __init__(
            self,
            A1=None, A2=None, A3=None, A4=None,
            *,
            Y: Union[NDArray, tuple[NDArray, NDArray]],
            X_true: Optional[NDArray] = None,
        ) -> None:
        if (A3 is None) != (A4 is None):
            raise ValueError("A3 and A4 must be both given or both None")
        if A1 is None and A2 is None and A3 is None:
            raise ValueError("At least one of A1, A2, A3 A4 must be given")
        self.A1, self.A2, self.A3, self.A4 = A1, A2, A3, A4

        if isinstance(Y, tuple):
            self.Yl, self.Yr = Y
        else:
            self.Yl, self.Yr = Y, None

        self.X_true = X_true

        self.error = []     # ||X_k - X_true||_F / ||X_true||_F per iteration
        self.residual = []  # ||R_k||_F per iteration
        self.niter = 0

    # ------------------------------------------------------------------ #
    # Factored applications of L, G and Y                                 #
    # ------------------------------------------------------------------ #

    def L_dot(self, Xl: NDArray, Xr: NDArray, Z: NDArray) -> NDArray:
        """Compute L[Xl Xr^T] @ Z without forming the n x n product."""
        XtZ = Xr.T @ Z
        out = np.zeros((Xl.shape[0], Z.shape[1]))
        if self.A1 is not None:
            out += self.A1 @ (Xl @ XtZ)
        if self.A2 is not None:
            out += Xl @ (Xr.T @ (self.A2 @ Z))
        if self.A3 is not None:
            out += self.A3 @ (Xl @ (Xr.T @ (self.A4 @ Z)))
        return out

    def LT_dot(self, Xl: NDArray, Xr: NDArray, W: NDArray) -> NDArray:
        """Compute L[Xl Xr^T]^T @ W without forming the n x n product."""
        XtW = Xl.T @ W
        out = np.zeros((Xr.shape[0], W.shape[1]))
        if self.A1 is not None:
            out += Xr @ (Xl.T @ (self.A1.T @ W))
        if self.A2 is not None:
            out += self.A2.T @ (Xr @ XtW)
        if self.A3 is not None:
            out += self.A4.T @ (Xr @ (Xl.T @ (self.A3.T @ W)))
        return out

    def Y_dot(self, Z: NDArray) -> NDArray:
        """Compute Y @ Z."""
        if self.Yr is None:
            return self.Yl @ Z
        return self.Yl @ (self.Yr.T @ Z)

    def YT_dot(self, W: NDArray) -> NDArray:
        """Compute Y^T @ W."""
        if self.Yr is None:
            return self.Yl.T @ W
        return self.Yr @ (self.Yl.T @ W)

    def G_dot(self, US: NDArray, V: NDArray, Z: NDArray) -> NDArray:
        """Defect G[X] @ Z = (L[X] - Y) @ Z for X = US V^T."""
        return self.L_dot(US, V, Z) - self.Y_dot(Z)

    def GT_dot(self, US: NDArray, V: NDArray, W: NDArray) -> NDArray:
        """Defect transposed, G[X]^T @ W."""
        return self.LT_dot(US, V, W) - self.YT_dot(W)

    # ------------------------------------------------------------------ #
    # Solver                                                              #
    # ------------------------------------------------------------------ #

    def solve(
            self,
            r: int,
            X0: Optional[tuple[NDArray, NDArray, NDArray]] = None,
            *,
            n: Optional[int] = None,
            tol: float = 1e-8,
            max_iter: int = 500,
            truncate_tol: float = 0.0,
            seed: Optional[int] = None,
            verbose: bool = True,
        ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Run the iteration; returns the factors (U, S, V) with X = U S V^T.

        r            : Rank of the iterate (fixed unless truncate_tol drops it).
        X0           : Initial factors (U0, S0, V0), U0/V0 orthonormal columns,
                       S0 an (r, r) matrix. Random rank-r start if None (needs n).
        n            : Problem dimension; only used when X0 is None.
        tol          : Stop when ||R_k||_F <= tol.
        max_iter     : Maximum number of iterations.
        truncate_tol : Relative threshold on the singular values of the
                       retracted core; 0.0 keeps the rank fixed at r.
        seed         : RNG seed for the random initial guess.
        verbose      : Print per-iteration residuals.
        """
        if X0 is not None:
            U, S, V = X0
        else:
            if n is None:
                raise ValueError("Provide either X0 or the dimension n")
            rng = np.random.default_rng(seed)
            U, _ = np.linalg.qr(rng.standard_normal((n, r)))
            V, _ = np.linalg.qr(rng.standard_normal((n, r)))
            S = np.diag(1e-3 * np.sort(rng.random(r))[::-1])

        self.error = []
        self.residual = []

        for k in range(max_iter):
            US = U @ S

            # -- Step 1: defect actions and projected residual ---------- #
            GV = self.G_dot(US, V, V)             # G[X] V,    (n, r)
            UtG = self.GT_dot(US, V, U).T         # U^T G[X],  (r, n)
            UtGV = UtG @ V                        # U^T G[X] V, (r, r)

            # -- Step 2: ||R||_F^2 = ||UtG||^2 + ||GV||^2 - ||UtGV||^2 -- #
            res2 = frobenius2(UtG) + frobenius2(GV) - frobenius2(UtGV)
            res = np.sqrt(max(res2, 0.0))
            self.residual.append(res)
            self._track_error(U, S, V)
            if verbose:
                print(f"iter {k:4d}   ||R||_F = {res:.6e}")
            if res <= tol:
                break

            # -- Steps 3-4: D = R, projected Hessian P(X) L[D] ---------- #
            # D = -(U UtG + Bp V^T) = Zl Zr^T,  Bp = (I - U U^T) G V
            Bp = GV - U @ UtGV
            Zl = np.hstack([U, Bp])
            Zr = np.hstack([-UtG.T, -V])

            HV = self.L_dot(Zl, Zr, V)            # L[D] V,   (n, r)
            UtH = self.LT_dot(Zl, Zr, U).T        # U^T L[D], (r, n)

            # -- Step 5: exact line search ------------------------------ #
            denom = (
                - inner_F(UtG, UtH)
                + inner_F(UtGV, UtH @ V)
                - inner_F(GV, HV)
            )
            alpha = res2 / denom

            # -- Step 6: DLRT update + retraction ----------------------- #
            K_new = US - alpha * GV
            L_new = V @ S.T - alpha * UtG.T
            U_hat, _ = np.linalg.qr(K_new)
            V_hat, _ = np.linalg.qr(L_new)

            G_Vhat = self.G_dot(US, V, V_hat)     # G[X] V_hat, (n, r)
            S_hat = (U_hat.T @ U) @ S @ (V.T @ V_hat) - alpha * (U_hat.T @ G_Vhat)

            Us, s, VsT = np.linalg.svd(S_hat)
            if truncate_tol > 0.0:
                keep = max(int(np.sum(s > truncate_tol * s[0])), 1)
                Us, s, VsT = Us[:, :keep], s[:keep], VsT[:keep, :]
            U = U_hat @ Us
            V = V_hat @ VsT.T
            S = np.diag(s)

        self.niter = len(self.residual) - 1
        return U, S, V

    def _track_error(self, U: NDArray, S: NDArray, V: NDArray) -> None:
        if self.X_true is not None:
            err = np.linalg.norm(U @ S @ V.T - self.X_true)
            self.error.append(err / np.linalg.norm(self.X_true))


class DynamicalLowRankSD(TikhonovSolver):
    """
    The note's Riemannian steepest descent / DLRT iteration applied to the
    inverse problem min Phi(X; y, w). Same interface as the other TikhonovSolver:

        solver = DynamicalLowRankSD(problem.operator, x_true=problem.x)
        x = solver.solve(problem.y_noisy, problem.weights, lambda_=1e-4)

    Differs from `DynamicalLowRankApproximation(method='sd')` in that the
    search direction is the *projected* (Riemannian) gradient R = -P(X)G with
    exact line search along it, and the retraction is the note's non-augmented
    K/L/S (DLRT) update.
    """

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            X0: str = 'qr',
            max_rank: int = 5,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01,
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} on the manifold of rank-`max_rank` matrices.

        y, NDArray          : The observed data (1D array).
        w, NDArray          : Tikhonov regularization weights (1D array).
        lambda_, float      : Tikhonov regularization parameter (squared internally).
        X0, str             : How to initialize X.
        max_rank, int       : Manifold rank r.
        max_iter, int       : Maximum number of iterations.
        rtol, float         : Stopping criterion on the relative Riemannian
                              gradient norm ||R_k|| / ||R_0||.
        etol, float         : Alternative stopping criterion, relative error.
                              Requires x_true to be set.
        seed, int|None      : Seed for random number generator (for initial X).
        verbose, bool       : Print out the results and progress.
        truncate_tol, float : Truncation tolerance used by the retraction.

        returns: Solution vector x = vec(X) (1D array).

        Note: `residual` tracks the *Riemannian* gradient norm — the natural
        stationarity measure on M_r.
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        # Initialize on the manifold at the rank-`max_rank` truncation of X0
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)
        Ux, Sx, Vx = self.truncate(Ux, Sx, Vx, truncate_tol, max_rank)
        X = Ux @ Sx @ Vx.T

        res0 = None
        err0 = (np.sqrt(frobenius2(X - self.X_true))
                if self._X_true is not None else None)

        for i in range(1, max_iter + 1):
            # -- Step 1: defect G = grad Phi and projected residual ------ #
            G = self.gradient(X, y, w, lambda_)
            GV = G @ Vx
            UtG = Ux.T @ G
            UtGV = UtG @ Vx

            # -- Step 2: ||R||^2 = ||UtG||^2 + ||GV||^2 - ||UtGV||^2 ----- #
            res2 = frobenius2(UtG) + frobenius2(GV) - frobenius2(UtGV)
            res = np.sqrt(max(res2, 0.0))
            if res0 is None:
                res0 = res
            rel_res = res / res0
            self.residual.append(rel_res)

            if err0 is not None:
                rel_err = np.sqrt(frobenius2(X - self.X_true)) / err0
                self.error.append(rel_err)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rtol criteria: rel_res={rel_res:.3}]")
                break
            if etol is not None and err0 is not None and rel_err < etol:
                if verbose: print(f"Converged at iter {i} [etol criteria: rel_err={rel_err:.3}]")
                break

            # -- Steps 3-4: D = R = -P(X)G, Hessian along D -------------- #
            D = -(Ux @ UtG + (GV - Ux @ UtGV) @ Vx.T)
            HD = self.apply_H(D, w, lambda_)

            # -- Step 5: exact line search (D tangent => <D,PHD> = <D,HD>) #
            alpha = res2 / inner_F(D, HD)

            # -- Step 6: DLRT update + retraction (non-augmented) -------- #
            K_new = Ux @ Sx - alpha * GV
            L_new = Vx @ Sx.T - alpha * UtG.T
            U_hat, _ = np.linalg.qr(K_new)
            V_hat, _ = np.linalg.qr(L_new)

            S_hat = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat) \
                    - alpha * (U_hat.T @ (G @ V_hat))
            Ux, Sx, Vx = self.truncate(U_hat, S_hat, V_hat, truncate_tol, max_rank)
            X = Ux @ Sx @ Vx.T

            if verbose and ((i % 100 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)


# ---------------------------------------------------------------------- #
# Demo: manufactured rank-r solution of a Sylvester-like equation          #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, r = 400, 5

    # Symmetric well-conditioned coefficients -> L self-adjoint, SPD
    lap = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)  # 1D Laplacian
    A1 = lap + np.eye(n)
    A2 = 0.5 * lap + np.eye(n)
    A3 = A4 = np.diag(1.0 + rng.random(n))

    # Manufactured rank-r ground truth and consistent right-hand side
    U_t, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V_t, _ = np.linalg.qr(rng.standard_normal((n, r)))
    S_t = np.diag(np.sort(rng.random(r))[::-1])
    X_true = U_t @ S_t @ V_t.T
    Y = A1 @ X_true + X_true @ A2 + A3 @ X_true @ A4

    solver = LowRankSylvesterSD(A1, A2, A3, A4, Y=Y, X_true=X_true)
    U, S, V = solver.solve(r, n=n, tol=1e-10, max_iter=2000, seed=1, verbose=False)

    print(f"iterations      : {solver.niter}")
    print(f"final residual  : {solver.residual[-1]:.3e}")
    print(f"relative error  : {solver.error[-1]:.3e}")
