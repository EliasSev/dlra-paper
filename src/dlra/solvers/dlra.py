"""
The dynamical low-rank solvers:

    - DynamicalLowRankApproximation : DLRA baseline (fixed / Adam / SD step rules)
    - DynamicalLowRankCG            : DLRA with conjugate search directions
    - DynamicalLowRankPCG           : preconditioned DLRA-CG

All subclass `DLRASolver`, which provides the "KLS" (basis-update & Galerkin)
integrator: a K/L/S step along the search direction followed by truncation back
to rank <= max_rank (`dlra_step`). The solvers differ in how they choose the
search direction D and the step size alpha.
"""
import numpy as np

from typing import Optional
from numpy.typing import NDArray

from dlra.solvers.base import TikhonovSolver, frobenius2, inner_F


class DLRASolver(TikhonovSolver):
    """
    Intermediate base for the solvers that advance the iterate X = Ux Sx Vx^T
    with the augmented K/L/S (basis-update & Galerkin) integrator. Subclasses
    choose a search direction D and a step size alpha, then call `dlra_step`.
    """

    def dlra_step(
            self,
            Ux: NDArray, Sx: NDArray, Vx: NDArray,
            D: NDArray, alpha: float,
            truncate_tol: float, max_rank: int,
        ) -> tuple[NDArray, NDArray, NDArray]:
        """One K/L/S step along direction D with step size alpha, then truncate."""
        # K-step
        K_star = (Ux @ Sx) + alpha * (D @ Vx)
        U_hat, _ = np.linalg.qr(np.hstack([Ux, K_star]))

        # L-step
        L_star = (Vx @ Sx.T) + alpha * (D.T @ Ux)
        V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

        # S-step
        S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
        S_new = S_new + alpha * (U_hat.T @ D @ V_hat)

        return self.truncate(U_hat, S_new, V_hat, truncate_tol, max_rank)


class DynamicalLowRankApproximation(DLRASolver):

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            method: str = 'adam',
            alpha: float = 0.1,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
            X0: str = 'qr',
            max_rank: int = 5,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01,
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLRA scheme
        with a selectable step rule.

        y, NDArray          : The observed data (1D array).
        w, NDArray          : Tikhonov regularization weights (1D array).
        lambda_, float      : Tikhonov regularization parameter (squared internally).
        method, str         : Step rule, one of:
                             'adam'  - adaptive step via Adam moments,
                             'fixed' - constant step size along -G,
                             'sd'    - steepest descent with exact line search.
        alpha, float        : Step size for 'fixed'; base learning rate for 'adam'.
                             Ignored when method='sd'.
        beta1, beta2, float : Adam moment decay rates. Used only when method='adam'.
        eps, float          : Adam denominator stabilizer. Used only when method='adam'.
        X0, str             : How to initialize X.
        max_rank, int       : Max rank of the solution (dynamical step).
        max_iter, int       : Maximum number of iterations.
        rtol, float         : Stopping criterion, relative residual (r0/rk).
        etol, float         : Alternative stopping criterion, relative error (e0/ek).
                              Requires x_true to be set.
        seed, int|None      : Seed for random number generator (for initial X).
        verbose, bool       : Print out the results and progress.
        truncate_tol, float : Truncation tolerance for the adaptive rank update.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        common = dict(
            X0=X0, max_rank=max_rank, max_iter=max_iter, rtol=rtol, etol=etol,
            seed=seed, verbose=verbose, truncate_tol=truncate_tol,
        )
        method = method.lower()
        if method == 'adam':
            return self._solve_adam(
                y, w, lambda_, alpha=alpha,
                beta1=beta1, beta2=beta2, eps=eps, **common,
            )
        elif method == 'fixed':
            return self._solve_fixed(y, w, lambda_, alpha=alpha, **common)
        elif method == 'sd':
            return self._solve_sd(y, w, lambda_, **common)
        else:
            raise ValueError(
                f"Unknown method: '{method}'. Use 'adam', 'fixed', or 'sd'."
            )

    def _solve_adam(
            self, y, w, lambda_, *,
            alpha, beta1, beta2, eps,
            X0, max_rank, max_iter, rtol, etol, seed, verbose, truncate_tol,
        ):
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = self._err0(X)

        m_D = np.zeros((self.n, self.n))
        v_D = np.zeros((self.n, self.n))

        for i in range(1, max_iter + 1):
            # Adam update for D
            m_D = beta1 * m_D + (1 - beta1) * D
            v_D = beta2 * v_D + (1 - beta2) * (D**2)
            m_hat = m_D / (1 - beta1**i)
            v_hat = v_D / (1 - beta2**i)
            D_adam = m_hat / (np.sqrt(v_hat) + eps)

            Ux, Sx, Vx = self.dlra_step(Ux, Sx, Vx, D_adam, alpha, truncate_tol, max_rank)

            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            D = -G.copy()

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def _solve_fixed(
            self, y, w, lambda_, *,
            alpha,
            X0, max_rank, max_iter, rtol, etol, seed, verbose, truncate_tol,
        ):
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = self._err0(X)

        for i in range(1, max_iter + 1):
            Ux, Sx, Vx = self.dlra_step(Ux, Sx, Vx, D, alpha, truncate_tol, max_rank)

            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            D = -G.copy()

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def _solve_sd(
            self, y, w, lambda_, *,
            X0, max_rank, max_iter, rtol, etol, seed, verbose, truncate_tol,
        ):
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = self._err0(X)

        for i in range(1, max_iter + 1):
            # Exact line search along D = -G:  alpha = ||G||^2 / <D, HD>
            HD = self.apply_H(D, w, lambda_)
            alpha = frobenius2(G) / inner_F(D, HD)

            Ux, Sx, Vx = self.dlra_step(Ux, Sx, Vx, D, alpha, truncate_tol, max_rank)

            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            D = -G.copy()

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)


class DynamicalLowRankCG(DLRASolver):
    """
    Dynamical Low-Rank Conjugate Gradient: the DLRA K/L/S step taken along
    CG search directions with exact line search on the quadratic.
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
            restart_every: Optional[int] = None,
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLR-CG scheme.

        y, NDArray              : The observed data (1D array).
        w, NDArray              : Tikhonov regularization weights (1D array).
        lambda_, float          : Tikhonov regularization parameter (squared internally).
        max_iter, int           : Maximum number of iterations.
        max_rank, int           : Max rank of the solution (dynamical step).
        rtol, float             : Stopping criterion, relative residual (r0/rk).
        etol, float             : Alternative stopping criterion, relative error (e0/ek).
                                  Requires x_true to be set.
        seed, int|None          : Seed for random number generator (for initial X).
        verbose, bool           : Print out the results and progress.
        truncate_tol, float     : Truncation tolerance for the adaptive rank update.
        restart_every, int|None : If set, recompute the true gradient at the truncated X
                                  and reset D = -G every this many iterations, correcting
                                  drift between the CG recurrence and the truncated iterate.
                                  If None, no restarting is performed.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        # Initialize X (random)
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        # Initial residual
        res0 = np.sqrt(frobenius2(G))
        err0 = self._err0(X)

        for i in range(1, max_iter + 1):
            # Step size
            HD = self.apply_H(D, w, lambda_)
            alpha = frobenius2(G) / inner_F(D, HD)

            # K/L/S step + truncation
            Ux, Sx, Vx = self.dlra_step(Ux, Sx, Vx, D, alpha, truncate_tol, max_rank)
            X = Ux @ Sx @ Vx.T

            # Update the gradient and the search direction
            if restart_every is not None and i % restart_every == 0:
                G = self.gradient(X, y, w, lambda_)
                D = -G.copy()
            else:
                denom = frobenius2(G)
                G = G + alpha * HD
                beta = frobenius2(G) / denom
                D = -G + beta * D

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(X)


class DynamicalLowRankPCG(DLRASolver):
    """
    Dynamical Low-Rank Preconditioned Conjugate Gradient.
    """

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            max_rank: int = 5,
            *,
            preconditioner: str = 'ic',
            truncate_tol: float = 0.01,
            X0: str = 'qr',
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            restart_every: Optional[int] = None,
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLR-PCG scheme.
        The preconditioner must be one of: 'none', 'jacobi', 'ssor', 'ic', 'ic-woodbury', 'perfect'

        y, NDArray          : The observed data (1D array).
        w, NDArray          : Tikhonov regularization weights (1D array).
        lambda_, float      : Tikhonov regularization parameter (squared internally).
        max_iter, int       : Maximum number of iterations.
        max_rank, int       : Max rank of the solution (dynamical step).
        preconditioner, str : Preconditioner to be used (P^{-1}).
        rtol, float         : Stopping criterion, relative residual (r0/rk).
        etol, float|None    : Alternative stopping criterion, relative error.
                              Requires x_true to be set.
        seed, int|None      : Seed for random number generator (for initial X).
        verbose, bool       : Print out the results and progress.
        truncate_tol, float : Truncation tolerance for the adaptive rank update.
        X0, str             : How to initialize X.
        restart_every, int|None : If set, recompute the true gradient at the truncated X
                                  and reset D = -G every this many iterations.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        # Initialize X (random)
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank, X0)

        # Preconditioner (1d np.array or LinearOperator)
        P_inv = self.get_preconditioner(w, lambda_, preconditioner)

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        Z = self.apply_P_inv(G, P_inv)
        D = -Z.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = self._err0(X)

        for i in range(1, max_iter + 1):
            # Step size
            HD = self.apply_H(D, w, lambda_)
            denom = inner_F(D, HD)
            alpha = inner_F(G, Z) / denom

            # K/L/S step + truncation
            Ux, Sx, Vx = self.dlra_step(Ux, Sx, Vx, D, alpha, truncate_tol, max_rank)
            X = Ux @ Sx @ Vx.T

            # Update the gradient and the search direction
            if restart_every is not None and i % restart_every == 0:
                G = self.gradient(X, y, w, lambda_)
                Z = self.apply_P_inv(G, P_inv)
                D = -Z.copy()
            else:
                denom = inner_F(G, Z)
                G = G + alpha * HD
                Z = self.apply_P_inv(G, P_inv)
                beta = inner_F(G, Z) / denom
                D = -Z + beta * D

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(X)
