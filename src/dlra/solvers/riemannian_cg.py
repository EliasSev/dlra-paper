"""
Riemannian conjugate gradient on the fixed-rank manifold M_r, following
`Riemannian_DLRA_CG.tex` ("Riemannian CG on M_r, factored").

Differs from `DynamicalLowRankCG` in three ways:

- the gradient is the *Riemannian* gradient P(X) grad Phi, recomputed at the
  true iterate every iteration -- there is no CG gradient recurrence, hence no
  drift between the recurrence and the truncated iterate (the `restart_every`
  motivation of DLRA-CG does not apply);
- all CG bookkeeping lives in tangent-space factors: a tangent vector at
  X = Ux Sx Vx^T is stored as the triple (M, Up, Vp) with

      xi = Ux M Vx^T + Up Vx^T + Ux Vp^T,   Ux^T Up = 0,  Vx^T Vp = 0,

  and the old direction/gradient are moved to the new tangent space by
  projection (vector transport), with a PR+ conjugacy coefficient and a
  descent guard;
- the data term B = mat(K^T M_ds y) enters through a truncated SVD
  B ~ Ub Sb Vb^T computed once per solve, so the gradient never forms the
  dense residual.

On the "no n x n objects" ambition: the update, transport, line search and
retraction all work on thin (n x r / n x 2r) factors. The one exception is the
Hessian oracle: H = K^T M_ds K + lambda^2 W^T M W acts on the *vectorized* DOF
ordering (see `matrix_to_vec`) and has no Kronecker structure, so `_riem_grad`
and `_qform` form one dense n x n intermediate per application and go through
`TikhonovSolver.apply_H` (two applications per iteration). Override those two
methods to plug in truly structured oracles.
"""
import numpy as np

from typing import Optional
from numpy.typing import NDArray

from dlra.io import progress_bar
from dlra.solvers.base import TikhonovSolver, frobenius2, inner_F


class RiemannianCG(TikhonovSolver):
    """
    Riemannian CG for min Phi(X; y, w) on the manifold of rank-`max_rank`
    matrices. Same interface as the other solvers:

        solver = RiemannianCG(problem.operator, x_true=problem.x)
        x = solver.solve(problem.y_noisy, problem.weights, lambda_=1e-4)
    """

    # ------------------------------------------------------------------ #
    # Tangent-space helpers: a tangent vector is the triple (M, Up, Vp)   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _t_inner(xi, eta) -> float:
        """Frobenius inner product of two tangent vectors at the same X."""
        return sum(inner_F(a, b) for a, b in zip(xi, eta))

    @staticmethod
    def _tangent_factors(Ux, Vx, xi):
        """
        Rank-2r factors (Fl, C, Fr) of the tangent vector xi = (M, Up, Vp),
        so that its dense form is Fl @ C @ Fr.T (never built here).
        """
        M, Up, Vp = xi
        I, O = np.eye(M.shape[0]), np.zeros_like(M)
        Fl = np.hstack([Ux, Up])
        Fr = np.hstack([Vx, Vp])
        C = np.block([[M, I], [I, O]])
        return Fl, C, Fr

    @staticmethod
    def _project(Fl, C, Fr, U, V):
        """
        Project the factored matrix A = Fl C Fr^T onto the tangent space at
        (U, V) via thin products only (vector transport).
        """
        Zv = Fl @ (C @ (Fr.T @ V))    # A V,   (n, r)
        Zu = Fr @ (C.T @ (Fl.T @ U))  # A^T U, (n, r)
        M = U.T @ Zv
        Up = Zv - U @ M
        Vp = Zu - V @ M.T
        return M, Up, Vp

    # ------------------------------------------------------------------ #
    # Hessian oracles (dense n x n intermediates; override for structured #
    # Hessians, see module docstring)                                     #
    # ------------------------------------------------------------------ #

    def _riem_grad(self, Ux, Sx, Vx, B, w, lambda_):
        """
        Riemannian gradient xi = P(X) grad Phi at X = Ux Sx Vx^T as tangent
        factors, with grad Phi = mat(H vec X) - B and B = (Ub, sb, Vb).
        Returns (xi, X); the dense X is reused by the caller for error
        tracking and the final solution.
        """
        X = Ux @ Sx @ Vx.T                 # dense intermediate for apply_H
        HX = self.apply_H(X, w, lambda_)
        Ub, sb, Vb = B
        Zv = HX @ Vx - Ub @ (sb[:, None] * (Vb.T @ Vx))    # G Vx
        Zu = HX.T @ Ux - Vb @ (sb[:, None] * (Ub.T @ Ux))  # G^T Ux
        M = Ux.T @ Zv
        Up = Zv - Ux @ M
        Vp = Zu - Vx @ M.T
        return (M, Up, Vp), X

    def _qform(self, Fl, C, Fr, w, lambda_):
        """Hessian quadratic form <A, mat(H vec A)>_F for A = Fl C Fr^T."""
        A = Fl @ C @ Fr.T                  # dense intermediate for apply_H
        return inner_F(A, self.apply_H(A, w, lambda_))

    # ------------------------------------------------------------------ #
    # Solver                                                              #
    # ------------------------------------------------------------------ #

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
            truncate_tol: float = 0.0,
            restart_every: Optional[int] = None,
            b_tol: float = 1e-12,
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} on the manifold of rank-`max_rank` matrices.

        y, NDArray              : The observed data (1D array).
        w, NDArray              : Tikhonov regularization weights (1D array).
        lambda_, float          : Tikhonov regularization parameter (squared internally).
        X0, str                 : How to initialize X.
        max_rank, int           : Manifold rank r.
        max_iter, int           : Maximum number of iterations.
        rtol, float             : Stopping criterion on the relative Riemannian
                                  gradient norm ||xi_k|| / ||xi_0||.
        etol, float             : Alternative stopping criterion, relative error.
                                  Requires x_true to be set.
        seed, int|None          : Seed for random number generator (for initial X).
        verbose, bool           : Print out the results and progress.
        truncate_tol, float     : Truncation tolerance used by the retraction;
                                  0.0 keeps the rank fixed at max_rank.
        restart_every, int|None : Periodic restart: reset the direction to the
                                  steepest descent -xi every this many iterations.
                                  (The descent guard resets automatically either way.)
        b_tol, float            : Relative tolerance for the truncated SVD of the
                                  data term B = mat(K^T M_ds y).

        returns: Solution vector x = vec(X) (1D array).

        Note: `residual` tracks the *Riemannian* gradient norm — the natural
        stationarity measure on M_r.
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        # Compressed data term B = mat(K^T M_ds y) ~ Ub diag(sb) Vb^T
        b = self.V @ (self.S * (self.UT @ (self.M_ds @ y)))
        Ub, sb, VbT = np.linalg.svd(self.vec_to_matrix(b), full_matrices=False)
        kb = max(int(np.sum(sb > sb[0] * b_tol)), 1)
        B = (Ub[:, :kb], sb[:kb], VbT[:kb, :].T)

        # Initialize on the manifold at the rank-`max_rank` truncation of X0
        _, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)
        Ux, Sx, Vx = self.truncate(Ux, Sx, Vx, truncate_tol, max_rank)

        # Initial Riemannian gradient, direction and residual
        xi, X = self._riem_grad(Ux, Sx, Vx, B, w, lambda_)
        eta = tuple(-a for a in xi)
        g = self._t_inner(xi, xi)

        res0 = np.sqrt(g)
        err0 = self._err0(X)

        for i in range(1, max_iter + 1):
            # Step size (exact quadratic line search along eta). Guard against
            # a numerically zero direction (xi at machine precision): the
            # curvature underflows and alpha becomes 0/0.
            Fl_e, C_e, Fr_e = self._tangent_factors(Ux, Vx, eta)
            xi_eta = self._t_inner(xi, eta)
            qf = self._qform(Fl_e, C_e, Fr_e, w, lambda_)
            if not np.isfinite(qf) or qf <= 0 or xi_eta >= 0:
                if verbose: print(f"Stagnated at iter {i} [<xi,eta>={xi_eta:.3}, <eta,H eta>={qf:.3}]")
                i -= 1
                break
            alpha = -xi_eta / qf

            # Retraction: X + alpha*eta in rank-2r factors, then truncate.
            # QR of the *stacked* factors (not of Up/Vp alone, as in the tex):
            # equivalent in exact arithmetic, but keeps U_hat/V_hat orthonormal
            # even when Up/Vp are numerically rank-deficient near convergence.
            M_e, Up_e, Vp_e = eta
            I = np.eye(M_e.shape[0])
            U_hat, Ru = np.linalg.qr(np.hstack([Ux, Up_e]))
            V_hat, Rv = np.linalg.qr(np.hstack([Vx, Vp_e]))
            C = np.block([
                [Sx + alpha * M_e, alpha * I],
                [alpha * I, np.zeros_like(M_e)],
            ])
            Ux_new, Sx_new, Vx_new = self.truncate(
                U_hat, Ru @ C @ Rv.T, V_hat, truncate_tol, max_rank
            )

            # New Riemannian gradient, always at the true iterate
            xi_new, X = self._riem_grad(Ux_new, Sx_new, Vx_new, B, w, lambda_)
            g_new = self._t_inner(xi_new, xi_new)

            # Track residual/error and check convergence
            rel_res = np.sqrt(g_new) / res0
            self.residual.append(rel_res)
            if err0 is not None:
                rel_err = np.sqrt(frobenius2(X - self.X_true)) / err0
                self.error.append(rel_err)

            if rel_res < rtol:
                Ux, Sx, Vx = Ux_new, Sx_new, Vx_new
                if verbose: print(f"Converged at iter {i} [rtol criteria: rel_res={rel_res:.3}]")
                break
            if etol is not None and err0 is not None and rel_err < etol:
                Ux, Sx, Vx = Ux_new, Sx_new, Vx_new
                if verbose: print(f"Converged at iter {i} [etol criteria: rel_err={rel_err:.3}]")
                break

            # Transport the old direction and gradient (both factored at the
            # old (Ux, Vx)) to the new tangent space
            eta_t = self._project(Fl_e, C_e, Fr_e, Ux_new, Vx_new)
            Fl_x, C_x, Fr_x = self._tangent_factors(Ux, Vx, xi)
            xi_t = self._project(Fl_x, C_x, Fr_x, Ux_new, Vx_new)

            # Conjugacy coefficient (PR+) and new direction
            beta = max(0.0, (g_new - self._t_inner(xi_new, xi_t)) / g)
            eta = tuple(-a + beta * b_ for a, b_ in zip(xi_new, eta_t))

            # Descent guard / periodic restart
            if (self._t_inner(xi_new, eta) >= 0
                    or (restart_every is not None and i % restart_every == 0)):
                eta = tuple(-a for a in xi_new)

            Ux, Sx, Vx = Ux_new, Sx_new, Vx_new
            xi, g = xi_new, g_new

            if verbose and ((i % 100 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        self.Ux, self.Sx, self.Vx = Ux, Sx, Vx
        return self.matrix_to_vec(X)
