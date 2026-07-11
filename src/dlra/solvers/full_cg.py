"""
Full-rank conjugate gradient on the discrete objective Phi -- the baseline the
low-rank solvers are validated against.
"""
import numpy as np

from typing import Optional
from numpy.typing import NDArray

from dlra.solvers.base import TikhonovSolver, frobenius2, inner_F


class ConjugateGradient(TikhonovSolver):

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            X0: str = 'qr',
            X0_rank: int = 3,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ using a standard CG scheme.

        y, NDArray     : The observed data (1D array).
        w, NDArray     : Tikhonov regularization weights (1D array).
        lambda_, float : Tikhonov regularization parameter (pass lambda, it is
                         squared internally).
        max_iter, int  : Maximum number of iterations.
        X0, str        : How to initialize X.
        X0_rank, int   : The rank of X0 (only used if X0 = 'low-rank-qr' or 'householder')
        rtol, float    : Stopping criterion, relative residual (r0/rk).
        etol, float    : Alternative stopping criterion, relative error (e0/ek).
        seed, int|None : Seed for random number generator (for initial X).
        verbose, bool  : Print out the results and progress.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        # Initialize X (random)
        X = self.initial_X(seed, max_rank=X0_rank, X0=X0)[0]

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

            # Update X
            X = X + alpha * D

            # Update the gradient and the search direction
            denom = frobenius2(G)
            G = G + alpha * HD
            beta = frobenius2(G) / denom
            D = -G + beta * D

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(X)
