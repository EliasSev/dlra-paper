# Context: DLRA-CG for a browser conversation

This is background for a new idea from my supervisor: can DLRA-CG be
implemented **without ever forming the full $n \times n$ matrix $X$**, working
only with its low-rank (SVD) factors $U_x, \Sigma_x, V_x$? I'm not interested
in rSVD or the SVD of the forward operator $K$ here — $K$ and the weights $W$
should just be treated as given operators. The focus is entirely on the
low-rank solver side.

## 1. The optimization problem

The underlying inverse problem, after FEM discretization, is a regularized
least-squares problem for a source vector $x \in \mathbb{R}^N$:

$$
\min_{x \in \mathbb{R}^N} \; \|Kx - y\|_{M_\partial}^2 + \lambda^2 \|Wx\|_M^2 .
$$

- $K \in \mathbb{R}^{N_b \times N}$ is the (discretized) forward operator mapping
  a volume source to boundary data.
- $y \in \mathbb{R}^{N_b}$ is the observed boundary data.
- $M$, $M_\partial$ are the volume/boundary FEM mass matrices (both SPD,
  sparse).
- $W$ is a diagonal regularization weight matrix.
- $\lambda > 0$ is the regularization parameter.

This is a quadratic objective $\Phi(x) = \tfrac12 \|Kx-y\|_{M_\partial}^2 +
\tfrac{\lambda^2}{2}\|Wx\|_M^2$, with gradient and Hessian

$$
\nabla \Phi(x) = K^T M_\partial (Kx - y) + \lambda^2 W^T M W x, \qquad
H = K^T M_\partial K + \lambda^2 W^T M W.
$$

$H$ is SPD. The minimizer solves the normal equations $Hx = K^T M_\partial y$.

**Reshaping to a matrix.** The source $x \in \mathbb{R}^{n^2}$ (for an $n\times
n$ mesh) is reshaped into a matrix $X = \operatorname{mat}(x) \in
\mathbb{R}^{n \times n}$. In many applications the source is a sum of a few
localized/sparse features, so $X$ is well approximated by a **low-rank**
matrix: $\operatorname{rank}(X) = r \ll n$. This motivates restricting the
search to the manifold of rank-$r$ matrices,

$$
\mathcal{M}_r = \{ X \in \mathbb{R}^{n\times n} : \operatorname{rank}(X) = r \},
$$

and working with a factorized representation $X = U_x \Sigma_x V_x^T$ ($U_x,
V_x$ orthonormal columns, $\Sigma_x$ nonsingular) instead of the dense $n
\times n$ matrix.

## 2. Dynamical low-rank approximation (DLRA)

A generic iterative update $X^{(i+1)} = X^{(i)} + \alpha_i D^{(i)}$ does not
preserve rank in general: adding an arbitrary direction $D^{(i)}$ to a rank-$r$
matrix typically produces a higher-rank result. DLRA (Koch–Lubich) fixes this
by projecting the update direction onto the **tangent space**
$\mathcal{T}_{X^{(i)}}(\mathcal{M}_r)$ of the manifold at the current iterate,

$$
X^{(i+1)} = X^{(i)} + \alpha_i \, \mathcal{P}_{X^{(i)}}\!\big(D^{(i)}\big).
$$

In this thesis/paper we use the discrete "KLS" (basis-update & Galerkin)
integrator of Schotthöfer et al., which works directly with the factors and
avoids ever forming $X$ explicitly:

**DLRA step**, given $X^{(i)} = U_x \Sigma_x V_x^T$, direction $D^{(i)}$, step
size $\alpha_i$:

$$
\begin{aligned}
K' &= U_x \Sigma_x + \alpha_i D^{(i)} V_x, &
L' &= V_x \Sigma_x^T + \alpha_i (D^{(i)})^T U_x, \\
\hat U &= \operatorname{orth}([U_x,\, K']), &
\hat V &= \operatorname{orth}([V_x,\, L']), \\
\hat \Sigma &= (\hat U^T U_x)\,\Sigma_x\,(V_x^T \hat V) + \alpha_i\, \hat U^T D^{(i)} \hat V, \\
(U_x, \Sigma_x, V_x) &= \operatorname{truncate}_r(\hat U, \hat\Sigma, \hat V).
\end{aligned}
$$

Here $[U_x, K']$ means column-concatenation, $\operatorname{orth}(\cdot)$
is a thin QR (or similar) orthogonalization, and $\operatorname{truncate}_r$
takes the SVD of the small $2r \times 2r$ matrix $\hat\Sigma$ and keeps the
top $r$ singular values/vectors, rotating them back into $\hat U, \hat V$.

**Key point for the new idea:** every quantity here — $U_x, \Sigma_x, V_x, K',
L', \hat U, \hat V, \hat\Sigma$ — has a dimension controlled by $r$ (or $2r$),
not by $n$. The only place $n$-sized objects appear is in evaluating the
*direction* $D^{(i)}$ (which requires the gradient $\nabla\Phi(X^{(i)})$, an $n
\times n$ object) and in the matrix products $U_x^T(\cdot)$, $(\cdot)V_x$ that
touch $D^{(i)}$.

## 3. DLRA-CG: conjugate gradient search directions

Since $\Phi$ is quadratic with SPD Hessian $H$, plain gradient descent
directions ($D^{(i)} = -\nabla\Phi(X^{(i)})$) can be replaced by
**$H$-conjugate** directions from the CG method,
$\langle D^{(i)}, H D^{(j)}\rangle_F = 0$ for $i \ne j$, which exploit
curvature information and converge much faster. This is the core idea of
**DLRA-CG**: combine the CG search-direction recurrence with the DLRA
tangent-projected update so the iterate $X^{(i)}$ stays on $\mathcal{M}_r$.

**DLRA-CG algorithm.** Initialize $G^{(0)} = \nabla\Phi(X^{(0)})$,
$D^{(0)} = -G^{(0)}$. For $i = 0, 1, 2, \dots$:

$$
\alpha_i = \frac{\|G^{(i)}\|_F^2}{\langle D^{(i)}, H D^{(i)} \rangle_F}
$$

then perform the DLRA (KLS) step above with this $\alpha_i$ and $D^{(i)}$ to
get $X^{(i+1)} = U_x \Sigma_x V_x^T$, then update the search direction using
the Fletcher–Reeves recurrence:

$$
G^{(i+1)} = G^{(i)} + \alpha_i\, H D^{(i)}, \qquad
\beta_i = \frac{\|G^{(i+1)}\|_F^2}{\|G^{(i)}\|_F^2}, \qquad
D^{(i+1)} = -G^{(i+1)} + \beta_i D^{(i)}.
$$

Note $HD^{(i)}$ here means $\operatorname{mat}[H\operatorname{vec}(D^{(i)})]$
— applying the (vectorized) Hessian to the matrix direction and reshaping back.

**Important subtlety (why $G^{(i)}$ drifts from the true gradient).** The CG
recurrence $G^{(i+1)} = G^{(i)} + \alpha_i H D^{(i)}$ is only mathematically
equal to the true gradient $\nabla\Phi(X^{(i+1)})$ if $X^{(i+1)} = X^{(i)} +
\alpha_i D^{(i)}$ *exactly*. But DLRA-CG *truncates* every step back onto
$\mathcal{M}_r$, so in reality $X^{(i+1)} = \operatorname{truncate}_r(X^{(i)} +
\alpha_i \mathcal P_{X^{(i)}}(D^{(i)})) \ne X^{(i)} + \alpha_i D^{(i)}$. The CG
recurrence is silently tracking the gradient along a *different*, full-rank,
never-materialized trajectory

$$
Y^{(i)} = X^{(0)} + \sum_{j<i} \alpha_j D^{(j)},
$$

not along the actual (truncated) manifold iterates $X^{(i)}$. As $Y^{(i)}$ and
$X^{(i)}$ drift apart, $G^{(i)} \to 0$ (plain CG converges along $Y^{(i)}$),
so $\alpha_i \to 0$ and $X^{(i)}$ can stall short of the true minimizer.
**Restarted CG** periodically recomputes the true gradient $G^{(i)} =
\nabla\Phi(X^{(i)})$ and resets $D^{(i)} = -G^{(i)}$, discarding the CG history
— this both counteracts loss of $H$-conjugacy from roundoff (the classical
reason to restart CG) and resynchronizes $Y^{(i)}$ back to $X^{(i)}$.

## 4. DLRA-PCG (preconditioned variant)

Preconditioning replaces $Hx = K^T M_\partial y$ by the equivalent system
$P^{-1}Hx = P^{-1}K^T M_\partial y$ for some $P \approx H$ with
$\kappa(P^{-1}H) \ll \kappa(H)$. Standard PCG modifications apply, with
$Z^{(i)} = P^{-1} G^{(i)}$:

$$
\alpha_i = \frac{\langle G^{(i)}, Z^{(i)} \rangle_F}{\langle D^{(i)}, H D^{(i)} \rangle_F}, \qquad
\beta_i = \frac{\langle G^{(i+1)}, Z^{(i+1)} \rangle_F}{\langle G^{(i)}, Z^{(i)} \rangle_F}, \qquad
D^{(i+1)} = -Z^{(i+1)} + \beta_i D^{(i)}.
$$

(The choice of $P$ — Jacobi, incomplete Cholesky, Woodbury-corrected, "perfect"
— is not relevant to the current question and is left out here; it depends on
structural properties of $H$, not on how $X$ is represented.)

## 5. Where the "form full $X$" problem actually lives

In the current implementation, $X^{(i)}$ is kept factored as $U_x, \Sigma_x,
V_x$ across iterations — the KLS step above never reconstructs
$U_x\Sigma_x V_x^T$ as a dense $n\times n$ matrix for the *update* itself. The
place where a full $n \times n$ (or its vectorized $N$-length form) object is
still unavoidable, as currently formulated, is the **gradient/direction
computation**:

$$
G^{(i)} = \nabla \Phi(X^{(i)}) = K^T M_\partial (K x^{(i)} - y) + \lambda^2 W^T M W x^{(i)}, \qquad x^{(i)} = \operatorname{vec}(X^{(i)}),
$$

and the Hessian application $H D^{(i)} = \operatorname{mat}[H
\operatorname{vec}(D^{(i)})]$, together with the CG bookkeeping quantities
$D^{(i)}$, $G^{(i)}$ themselves, which are generally *not* low rank (they are
sums/differences of matrices, gradients of the residual, etc.) — even though
$X^{(i)}$ stays exactly rank $r$.

**The new idea to explore:** can $G^{(i)}$, $D^{(i)}$, and the products used in
$\alpha_i$, $\beta_i$ (inner products like $\langle D^{(i)}, HD^{(i)}\rangle_F$,
$\|G^{(i)}\|_F^2$) all be expressed/approximated/maintained through low-rank
(SVD-like) factorizations of $G^{(i)}$ and $D^{(i)}$ themselves, so that no
step of the algorithm ever needs an $n\times n$ dense array — only thin
factors of size $n \times r'$ for some small (possibly growing) $r'$? This is
distinct from the existing DLRA truncation of $X^{(i)}$: it's about whether
the *auxiliary* CG quantities ($G^{(i)}$, $D^{(i)}$, and their Hessian
products) can also be kept low-rank throughout, rather than being formed
densely and only implicitly rank-limited through their effect on $X^{(i)}$.

## Notation summary

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^N$, $X = \operatorname{mat}(x) \in \mathbb{R}^{n\times n}$ | Source vector / reshaped matrix, $N = n^2$ |
| $r$ | Target solution rank, $\mathcal{M}_r$ = rank-$r$ manifold |
| $U_x, \Sigma_x, V_x$ | Low-rank factors of $X^{(i)}$, $X^{(i)} = U_x\Sigma_x V_x^T$ |
| $K$, $M$, $M_\partial$, $W$, $\lambda$ | Forward operator, mass matrices, weights, reg. parameter — all **given** |
| $\Phi$, $\nabla\Phi$, $H$ | Objective, gradient, Hessian ($H$ SPD) |
| $D^{(i)}$ | CG search direction (matrix-shaped) |
| $G^{(i)}$ | CG gradient tracker (not always $= \nabla\Phi(X^{(i)})$ exactly, see §3) |
| $\alpha_i$, $\beta_i$ | CG step size / conjugacy coefficient |
| $P$ | Preconditioner (DLRA-PCG only) |
| $\mathcal{P}_{X}(\cdot)$ | Projection onto tangent space $\mathcal{T}_X(\mathcal{M}_r)$ |
