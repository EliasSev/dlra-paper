"""
Preconditioners for the Hessian H = K^T M_ds K + lambda * W^T M W, exposed as a
1D diagonal (Jacobi) or a scipy LinearOperator applying P^{-1}. Used by
DynamicalLowRankPCG.

All builders receive `lambda_` *already squared* (the solvers square it once at
`solve` entry) and K's SVD factors: U (N_b, k), S (k,) 1D, V (N, k).
"""
import numpy as np
import scipy as sp

from numpy.typing import NDArray
from typing import Union

from pymatting import ichol  # incomplete Cholesky
from sksparse.cholmod import cholesky  # sparse Cholesky
from scipy.sparse.linalg import LinearOperator, factorized
from scipy.linalg import cho_factor, cho_solve

VALID_PRECONDITIONERS = ('none', 'jacobi', 'ssor', 'ic', 'ic-woodbury', 'perfect')


def get_preconditioner(
        name: str, *,
        U: NDArray, S: NDArray, V: NDArray,
        M_dx, M_ds,
        w: NDArray, lambda_: float,
    ) -> Union[NDArray, LinearOperator]:
    """Dispatch to a preconditioner builder by name."""
    name = name.lower()
    if name == 'none':
        return np.ones(len(w))
    elif name == 'jacobi':
        return jacobi(U, S, V, M_dx, M_ds, w, lambda_)
    elif name == 'ssor':
        return ssor(M_dx, w, lambda_)
    elif name == 'ic':
        return ic(M_dx, w, lambda_)
    elif name == 'ic-woodbury':
        return ic_woodbury(U, S, V, M_dx, M_ds, w, lambda_)
    elif name == 'perfect':
        return perfect(U, S, V, M_dx, M_ds, w, lambda_)
    else:
        raise ValueError(
            f"Unknown preconditioner '{name}'. Use one of: {VALID_PRECONDITIONERS}"
        )


def jacobi(U, S, V, M_dx, M_ds, w: NDArray, lambda_: float) -> NDArray:
    """
    Jacobi (diagonal) preconditioner: P = diag(H), returned as 1D array P^{-1}.
    """
    # diag(K^T M_ds K)
    A = U.T @ (M_ds @ U)                 # (k, k)
    B = (S[:, None] * A) * S             # (k, k)
    VB = V @ B                           # (N, k)
    diag_KtMK = np.sum(V * VB, axis=1)   # (N,)

    # diag(lambda * W^T M W)
    diag_M = M_dx.diagonal()             # (N,)
    diag_WtMW = lambda_ * w**2 * diag_M  # (N,)

    P = diag_KtMK + diag_WtMW
    return 1.0 / P


def ssor(M_dx, w: NDArray, lambda_: float, omega: float = 1.0) -> LinearOperator:
    """
    SSOR preconditioner applied to the sparse part S = lambda * W^T M W.
    For omega=1 this reduces to Symmetric Gauss-Seidel.
    P = (D/omega + L) (D/omega)^{-1} (D/omega + L^T),
    where D = diag(S) and L is the strictly lower triangular part of S.
    Applied as P^{-1} v via one forward and one backward triangular solve.
    """
    w_sp = sp.sparse.diags(w)
    S = lambda_ * w_sp @ M_dx @ w_sp

    D_over_omega = S.diagonal() / omega
    D_diag = sp.sparse.diags(D_over_omega)
    L = sp.sparse.tril(S, k=-1)   # strictly lower triangular

    lower = (D_diag + L).tocsc()
    upper = lower.T.tocsc()

    solve_lower = factorized(lower)
    solve_upper = factorized(upper)

    def apply_P_inv(v):
        y = solve_lower(v)
        z = D_over_omega * y
        return solve_upper(z)

    N = len(w)
    return LinearOperator((N, N), matvec=apply_P_inv)


def ic(M_dx, w: NDArray, lambda_: float) -> LinearOperator:
    """
    Simple IC preconditioner using only the sparse part of H:
    P approx S = lambda * W^T M W approx L L^T, applied as P^{-1} = L^{-T} L^{-1}.
    The low-rank part K^T M_ds K is ignored -- no Woodbury correction.
    """
    w_sp = sp.sparse.diags(w)
    S = lambda_ * w_sp @ M_dx @ w_sp
    S_csc = S.tocsc()

    L_ic = ichol(S_csc)

    def apply_P_inv(v):
        return L_ic(v)  # pymatting overloads __call__ to apply L^{-T} L^{-1}

    N = len(w)
    return LinearOperator((N, N), matvec=apply_P_inv)


def ic_woodbury(U, S, V, M_dx, M_ds, w: NDArray, lambda_: float) -> LinearOperator:
    """
    Woodbury-corrected IC preconditioner using the direct factored form:
      H = S_sp + U_tilde C U_tilde^T,
    where U_tilde = V_k Sigma_k (N x k) and C = U_k^T M_ds U_k (k x k).
    Applies the Woodbury identity directly without factoring C, giving:
      P^{-1} v = S^{-1}v - (S^{-1} U_tilde)(C^{-1} + U_tilde^T S^{-1} U_tilde)^{-1}(U_tilde^T S^{-1} v)
    where S^{-1} is approximated via incomplete Cholesky.
    """
    w_sp = sp.sparse.diags(w)
    S_sp = lambda_ * w_sp @ M_dx @ w_sp
    S_csc = S_sp.tocsc()

    U_tilde = V * S[None, :]              # (N, k): V_k Sigma_k
    C = U.T @ (M_ds @ U)                  # (k, k): U_k^T M_ds U_k
    C_inv = np.linalg.inv(C)

    L_ic = ichol(S_csc)

    def apply_Sinv(v):
        return L_ic(v)

    Sinv_Ut = np.column_stack(
        [apply_Sinv(U_tilde[:, i]) for i in range(U_tilde.shape[1])]
    )                                     # (N, k): S^{-1} U_tilde
    W_mat = C_inv + U_tilde.T @ Sinv_Ut   # (k, k): Woodbury correction matrix
    W_chol = cho_factor(W_mat)

    def apply_P_inv(v):
        Sinv_v = apply_Sinv(v)
        correction = Sinv_Ut @ cho_solve(W_chol, U_tilde.T @ Sinv_v)
        return Sinv_v - correction

    N = len(w)
    return LinearOperator((N, N), matvec=apply_P_inv)


def perfect(U, S, V, M_dx, M_ds, w: NDArray, lambda_: float) -> LinearOperator:
    """
    Woodbury-corrected preconditioner with an *exact* sparse Cholesky of
    S = lambda * W^T M W (no incomplete approximation):
      H = V Sigma U^T M_ds U Sigma V^T + lambda * diag(w) M diag(w)
    """
    w_sp = sp.sparse.diags(w)
    S_sp = lambda_ * w_sp @ M_dx @ w_sp
    S_csc = S_sp.tocsc()

    A = U.T @ (M_ds @ U)
    L_A = np.linalg.cholesky(A)
    F = V * S[None, :] @ L_A

    # Sparse Cholesky
    factor = cholesky(S_csc)

    # Precompute these once
    Sinv_F = factor.solve_A(F)            # (N, k)
    C = np.eye(len(S)) + F.T @ Sinv_F     # (k, k)
    C_chol = cho_factor(C)

    def apply_P_inv(v):
        Sinv_v = factor.solve_A(v)
        correction = Sinv_F @ cho_solve(C_chol, Sinv_F.T @ v)
        return Sinv_v - correction

    N = len(w)
    return LinearOperator((N, N), matvec=apply_P_inv)
