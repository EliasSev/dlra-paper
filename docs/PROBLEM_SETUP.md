# Problem setup: DLRA-CG for an elliptic source-identification problem

This document is the self-contained mathematical reference for the paper project.
It defines the inverse problem, the regularization weights, and the DLRA /
DLRA-CG algorithms, with the exact formulas and their correspondence to the code
in `src/`.

**Scope.** The paper focuses on the DLRA-CG solver. The elliptic inverse problem
is the motivating application. The forward operator $K$ and the weights $W$ are
treated as *given*; the randomized-SVD construction of $K$ and the matrix-free
approximation of the weights are out of scope (see [SCOPE](#scope-boundary)).

---

## 1. The forward problem

Let $\Omega \subset \mathbb{R}^2$ be a bounded domain with boundary
$\partial\Omega$. The state $u$ solves the diffusion–reaction problem with a
homogeneous Neumann condition,

$$
-\nabla\cdot(\sigma\nabla u) + c\,u = f \quad\text{in }\Omega,
\qquad
\frac{\partial u}{\partial n} = 0 \quad\text{on }\partial\Omega,
$$

with diffusion coefficient $\sigma(x) > 0$ (scalar, or symmetric positive
definite matrix-valued) and reaction coefficient $c(x) > 0$. The source $f$ is
the unknown.

The PDE defines the solution operator $\mathcal{S}: L^2(\Omega)\to H^1(\Omega)$,
$\mathcal{S}f = u$. The trace operator $\mathcal{T}: H^1(\Omega)\to
L^2(\partial\Omega)$ restricts to the boundary, $\mathcal{T}u = u|_{\partial\Omega}$.
The **forward operator** is their composition,

$$
\mathcal{K} = \mathcal{T}\circ\mathcal{S},
\qquad
\mathcal{K}f = u|_{\partial\Omega}.
$$

**Inverse problem.** Given boundary observations $y = u|_{\partial\Omega}$,
recover the interior source $f$ such that $\mathcal{K}f = y$.

**Ill-posedness.** $\mathcal{K}$ has a non-trivial null space (interior sources
invisible on the boundary), so the solution is non-unique; and $\mathcal{K}$ is
compact with rapidly decaying singular values, so the reconstruction is unstable
under noise. Both are why regularization is required.

---

## 2. Finite-element discretization

Triangulate $\Omega$ with $N$ nodes, $N_b$ of them on the boundary. Use the
piecewise-linear FE space $V_h = \operatorname{span}\{\phi_1,\dots,\phi_N\}$.
The weak form of the PDE, tested against the basis, gives

$$
A_h\,u = M\,x,
$$

where $A_h\in\mathbb{R}^{N\times N}$ is the stiffness matrix and
$M\in\mathbb{R}^{N\times N}$ the (volume) mass matrix, both symmetric positive
definite. Here $x$ is the coefficient vector of $f_h = \sum_j x_j\phi_j$ and $u$
the coefficient vector of the solution. The discrete solution operator is
$S = A_h^{-1}M$.

The trace is a selection matrix $T\in\mathbb{R}^{N_b\times N}$ picking boundary
coefficients. The **discrete forward operator** is

$$
K = T\,S = T\,A_h^{-1}M \in \mathbb{R}^{N_b\times N}.
$$

### Inner products

The FE discretization carries weighted inner products, *not* the Euclidean one:

$$
\langle x, y\rangle_M = x^\top M y \ \ (\text{on }\mathbb{R}^N),
\qquad
\langle u, v\rangle_{M_\partial} = u^\top M_\partial v \ \ (\text{on }\mathbb{R}^{N_b}),
$$

with $M_\partial\in\mathbb{R}^{N_b\times N_b}$ the boundary mass matrix. Norms
$\|x\|_M=\sqrt{x^\top M x}$, $\|u\|_{M_\partial}=\sqrt{u^\top M_\partial u}$.

### Adjoint

Under these inner products the discrete adjoint $K^*\in\mathbb{R}^{N\times N_b}$
is defined by $(Kx)^\top M_\partial y = x^\top M K^* y$ for all $x,y$, giving

$$
K^* = A_h^{-1} T^\top M_\partial.
$$

> **Note.** $K^*$ is *not* the matrix transpose $K^\top$. They are related by
> $K^\top M_\partial = M K^*$. The Euclidean gradient below is written with
> $K^\top M_\partial$, which is what the code applies.

---

## 3. The forward operator in low-rank form

$K$ has fast-decaying singular values, so it is used throughout in a rank-$k$
factored form

$$
K \approx U_k \Sigma_k V_k^\top,
\qquad
U_k\in\mathbb{R}^{N_b\times k},\ \Sigma_k\in\mathbb{R}^{k\times k},\ V_k\in\mathbb{R}^{N\times k}.
$$

Every operator application reduces to small matmuls (cost $O((N+N_b)k)$):

$$
Kx = U_k\big(\Sigma_k(V_k^\top x)\big),
\qquad
K^\top M_\partial r = V_k\big(\Sigma_k(U_k^\top M_\partial r)\big).
$$

This factorization is what makes the gradient, Hessian-vector product, and
preconditioners cheap. **How $K$'s factors are obtained is out of scope for the
paper.** Treat $U_k,\Sigma_k,V_k$ as given (in the code they come from a
matrix-free rSVD, but the paper does not foreground this).

---

## 4. Regularization: the weights

Standard Tikhonov ($W = I$) fails here: because $K$ has a large null space, its
solutions pile up near the boundary regardless of where the true source sits.

**Elvetun–Nielsen weights.** Write $\hat P = K^+K$, the orthogonal projection
onto $\operatorname{Nul}(K)^\perp$ (the observable / row space of $K$). Interior
basis functions have small $\|\hat P\phi_i\|$ and are therefore biased toward
zero in any minimum-norm solution. The fix rescales each basis function so its
projected norm is one. With the FE normalization $\|\phi_i\|_M=\sqrt{M_{ii}}$ and
the SVD $K = U\Sigma V^\top$,

$$
w_i = \frac{\|K^+K\,e_i\|_M}{\|\phi_i\|_M}
    = \frac{\|V V^\top e_i\|_M}{\sqrt{M_{ii}}},
\qquad i = 1,\dots,N,
\qquad
W = \operatorname{diag}(w_1,\dots,w_N).
$$

In practice $K^+$ amplifies noise in the small singular values, so the truncated
projection $P_k = K_k^+K = V_k V_k^\top$ (top-$k$ right singular vectors) is used.

> **Out of scope.** Any matrix-free / randomized approximation of these weights
> (e.g. an $M$-orthogonal projection estimate) is not part of the paper. Present
> $W$ as given, computed exactly from the SVD of $K$ as above.

**Transformed formulation (context only).** The weights are defined in
$\operatorname{Nul}(K)^\perp\subset\mathbb{R}^N$, but the residual $Kx-y$ lives in
$\mathbb{R}^{N_b}$. Elvetun–Nielsen reformulate $Kx=y$ as $K^+Kx = K^+y$ so the
residual lives in the same subspace as the weights. The DLRA-CG solver in this
project uses the **standard** (non-transformed) objective below; the transformed
version is background for why the weights are the right regularizer.

---

## 5. The discrete inverse problem and its objective

The regularized problem is

$$
\min_{x\in\mathbb{R}^N}\ \|Kx - y\|_{M_\partial}^2 + \lambda^2\|Wx\|_M^2,
$$

with regularization parameter $\lambda \ge 0$. Define the objective

$$
\Phi(X) = \tfrac12\|Kx - y\|_{M_\partial}^2 + \tfrac{\lambda^2}{2}\|Wx\|_M^2,
$$

where $X = \operatorname{mat}(x)\in\mathbb{R}^{n\times n}$ is the source reshaped
into a matrix (with $N = n^2$). Its Euclidean gradient and Hessian in $x$ are

$$
\nabla\Phi = K^\top M_\partial(Kx - y) + \lambda^2 W^\top M W x,
\qquad
H = K^\top M_\partial K + \lambda^2 W^\top M W.
$$

$H$ is SPD ($K^\top M_\partial K$ is PSD, $\lambda^2 W^\top M W$ is SPD). The
minimizer solves the **normal equations**

$$
H x = K^\top M_\partial y.
$$

Because $\Phi$ is quadratic with an SPD Hessian, conjugate gradient is the
natural solver, and the entry point for DLRA-CG.

### Low-rank prior

The sources of interest are sparse / localized, hence well approximated by
low-rank matrices $X = \sum_{i=1}^r \sigma_i u_i v_i^\top$ with $r\ll n$. The
methods restrict the search to the manifold $\mathcal{M}_r$ of rank-$r$ matrices,
working with the factors $u_i, v_i$ rather than the full $X$.

> **Subscript convention.** $X = U_x\Sigma_x V_x^\top$ denotes the SVD factors of
> the *iterate*; $K = U\Sigma V^\top$ denotes the factors of the *operator*. Keep
> the $x$ subscript to avoid collision.

---

## 6. DLRA (baseline)

The dynamical low-rank approximation keeps iterates on $\mathcal{M}_r$ by
projecting each update direction onto the tangent space $\mathcal{T}_{X}\mathcal{M}_r$:

$$
X^{(i+1)} = X^{(i)} + \alpha_i\,\mathcal{P}_{X^{(i)}}(D^{(i)}).
$$

Rather than forming the projection explicitly, the scheme uses the discrete
**basis-update–Galerkin (KLS) step** of the unconventional integrator
(Ceruti–Lubich; Schotthöfer et al.). Given $X^{(i)} = U_x\Sigma_x V_x^\top$ and a
direction $D^{(i)}$ (baseline DLRA uses $D^{(i)} = -\nabla\Phi(X^{(i)})$):

$$
\begin{aligned}
K' &= U_x\Sigma_x + \alpha_i D^{(i)} V_x, &\quad \hat U &= \operatorname{orth}([U_x,\ K']),\\
L' &= V_x\Sigma_x^\top + \alpha_i (D^{(i)})^\top U_x, &\quad \hat V &= \operatorname{orth}([V_x,\ L']),\\
\hat\Sigma &= (\hat U^\top U_x)\Sigma_x(V_x^\top\hat V) + \alpha_i(\hat U^\top D^{(i)}\hat V), &&\\
(U_x,\Sigma_x,V_x) &= \operatorname{truncate}_r(\hat U,\hat\Sigma,\hat V). &&
\end{aligned}
$$

The augmented factors have $2r$ columns; truncation brings them back to rank $r$
(optionally rank-adaptive via a threshold $\vartheta$).

**Step size.** For the quadratic $\Phi$, steepest descent uses the exact line
search along $D^{(i)} = -\nabla\Phi(X^{(i)})$,

$$
\alpha_i = \frac{\|\nabla\Phi(X^{(i)})\|_F^2}{\langle D^{(i)}, H D^{(i)}\rangle_F}.
$$

DLRA with this rule is slow; it is the baseline that DLRA-CG improves on.

---

## 7. DLRA-CG (the method)

**Idea.** Replace the steepest-descent directions of DLRA with $H$-conjugate CG
directions, so the search exploits curvature of $H$ and converges in far fewer
iterations.

CG builds directions $\{D^{(i)}\}$ with $\langle D^{(i)}, H D^{(j)}\rangle_F = 0$
for $i\ne j$. Starting from $G^{(0)} = \nabla\Phi(X^{(0)})$, $D^{(0)} = -G^{(0)}$,
each iteration does:

$$
\alpha_i = \frac{\|G^{(i)}\|_F^2}{\langle D^{(i)}, H D^{(i)}\rangle_F},
$$

then the **KLS low-rank update** (§6 above) along $D^{(i)}$ with step $\alpha_i$,
followed by the CG recurrence

$$
G^{(i+1)} = G^{(i)} + \alpha_i H D^{(i)},
\qquad
\beta_i = \frac{\|G^{(i+1)}\|_F^2}{\|G^{(i)}\|_F^2}\ \text{(Fletcher–Reeves)},
\qquad
D^{(i+1)} = -G^{(i+1)} + \beta_i D^{(i)}.
$$

Here $HD^{(i)} = \operatorname{mat}[H\operatorname{vec}(D^{(i)})]$, evaluated
through the low-rank factors of $K$.

### The full-rank drift and restarts

This is the subtle part; state it clearly in the paper.

The CG pair $(G^{(i)}, D^{(i)})$ is generated by the recurrence
$G^{(i+1)} = G^{(i)} + \alpha_i H D^{(i)}$, **independently of the truncated
manifold iterates** $X^{(i)}$. After the first step $G^{(i)}$ no longer equals
$\nabla\Phi(X^{(i)})$; instead it tracks the gradient along the *full-rank*
trajectory

$$
Y^{(i)} = X^{(0)} + \sum_{j<i}\alpha_j D^{(j)}.
$$

As plain CG converges along $Y^{(i)}$, $G^{(i)} = \nabla\Phi(Y^{(i)})\to 0$, so
$\alpha_i\to 0$ and the manifold iterate $X^{(i)}$ can **stall** at a suboptimal
point as $Y^{(i)}$ and $X^{(i)}$ drift apart.

**Restarted CG** (period $N_r$): every $N_r$ iterations, recompute the true
gradient $G^{(i)} = \nabla\Phi(X^{(i)})$ at the truncated iterate and reset
$D^{(i)} = -G^{(i)}$, discarding previous directions. This resets $Y^{(i)} =
X^{(i)}$, giving the CG recurrence a fresh start from the actual iterate. It also
cleans up loss of $H$-conjugacy from round-off.

---

## 8. DLRA-PCG (preconditioned variant)

Precondition the normal equations, $P^{-1}Hx = P^{-1}K^\top M_\partial y$ with
$P\approx H$. With $Z^{(i)} = P^{-1}G^{(i)}$ the recurrences become

$$
\alpha_i = \frac{\langle G^{(i)}, Z^{(i)}\rangle_F}{\langle D^{(i)}, H D^{(i)}\rangle_F},
\qquad
\beta_i = \frac{\langle G^{(i+1)}, Z^{(i+1)}\rangle_F}{\langle G^{(i)}, Z^{(i)}\rangle_F},
\qquad
D^{(i+1)} = -Z^{(i+1)} + \beta_i D^{(i)},
$$

with the same KLS low-rank update in between. This is standard PCG
(Concus–Golub–O'Leary) wrapped around the low-rank truncation.

### Structure of $H$ and the preconditioners

With $K$ of rank $k$, the Hessian splits into a low-rank plus a sparse part,

$$
H = \underbrace{K^\top M_\partial K}_{\text{rank }k\ =:\,F} + \underbrace{\lambda^2 W^\top M W}_{\text{sparse}\ =:\,S}.
$$

Writing $F = \tilde V C \tilde V^\top$ with $\tilde V = V\Sigma$ and
$C = U^\top M_\partial U\in\mathbb{R}^{k\times k}$ (SPD), the paper considers:

| Preconditioner | $P^{-1}$ | Notes |
|---|---|---|
| **Jacobi** | $\operatorname{diag}(H)^{-1}$ | cheapest; good if $H$ diagonally dominant |
| **IC** | $L^{-\top}L^{-1}$, $S\approx LL^\top$ | incomplete Cholesky of sparse part only; ignores $F$ |
| **IC + Woodbury** | IC of $S$ + Woodbury correction for $F$ | $(S+\tilde V C\tilde V^\top)^{-1} = S^{-1} - S^{-1}\tilde V(C^{-1}+\tilde V^\top S^{-1}\tilde V)^{-1}\tilde V^\top S^{-1}$ |
| **Perfect** | $H^{-1}$ | exact sparse Cholesky of $S$ + Woodbury; $P^{-1}=H^{-1}$. In plain PCG this solves in one step, but the per-iteration truncation means DLRA-PCG still needs several. |

---

## 9. Notation

| Symbol | Shape | Meaning |
|---|---|---|
| $\Omega,\ \partial\Omega$ | — | domain, boundary |
| $\sigma,\ c$ | — | diffusion, reaction coefficients |
| $f,\ x = \operatorname{vec}(X)$ | $N$ | source; coefficient vector / matrix $X\in\mathbb{R}^{n\times n}$, $N=n^2$ |
| $y$ | $N_b$ | boundary observations |
| $N,\ N_b$ | — | volume DOFs, boundary DOFs |
| $A_h,\ M$ | $N\times N$ | stiffness, volume mass matrix |
| $M_\partial$ | $N_b\times N_b$ | boundary mass matrix |
| $K = TA_h^{-1}M$ | $N_b\times N$ | forward operator |
| $K^* = A_h^{-1}T^\top M_\partial$ | $N\times N_b$ | adjoint (note $K^*\ne K^\top$) |
| $U_k,\Sigma_k,V_k$ | — | rank-$k$ SVD factors of $K$ |
| $U_x,\Sigma_x,V_x$ | — | SVD factors of the iterate $X$ |
| $W = \operatorname{diag}(w)$ | $N\times N$ | Elvetun–Nielsen weights |
| $\lambda$ | — | regularization parameter |
| $\Phi,\ \nabla\Phi,\ H$ | — | objective, gradient, Hessian |
| $\mathcal{M}_r$ | — | manifold of rank-$r$ matrices |
| $r$ | — | solution / manifold rank |
| $k$ | — | operator rank (rank of $K$) |
| $N_r$ | — | CG restart period |
| $P$ | $N\times N$ | preconditioner |

---

## 10. Correspondence to the reference code {#code}

The reference implementation lives in `cg_solvers.py` (class `DynamicalLowRankCG`,
and `DynamicalLowRankPCG`). Points where the code and the formulas differ in
presentation:

- **$\lambda$ is squared inside `solve`.** The methods run `lambda_ = lambda_**2`
  at entry, so the argument you pass is $\lambda$ and the term applied is
  $\lambda^2 W^\top M W$.
- **$K$ enters only through its factors.** `gradient` and `apply_H` never form
  $K$: they compute $Kx = U(\Sigma(V^\top x))$ and
  $K^\top M_\partial r = V(\Sigma(U^\top M_\partial r))$ from `self.U, self.S,
  self.VT`. The solver is constructed from a trained rSVD object that supplies
  these factors (the one dependency on the out-of-scope rSVD machinery).
- **$W$ diagonal.** $W^\top M W x = w\odot(M(w\odot x))$ with $w$ a 1D array.
- **vec/mat ordering is not a plain reshape.** `matrix_to_vec` / `vec_to_matrix`
  permute via a lexsort of the FE DOF coordinates (`grid_indices`,
  `dof_indices`), so $\operatorname{mat}(\cdot)$ maps DOF order to a visual
  $n\times n$ grid. Reuse this permutation; do not assume `x.reshape(n,n)`.
- **`truncate(U, S, V, tol, max_rank)`** re-SVDs the small core $S$, applies a
  tail-sum tolerance test, then caps the rank at `max_rank`.
- **`restart_every`** implements the restarted-CG logic of §7 above.
- **Convergence tracking:** `.residual` (relative gradient norm) and `.error`
  (relative Frobenius error vs. `x_true`) are populated during `solve`; the
  return value is the flat solution vector.

Sibling solvers, useful as baselines/ablations: `ConjugateGradient` (full-rank
CG reference) and `DynamicalLowRankApproximation` (baseline DLRA with
`sd`/`fixed`/`adam` steps).

---

## Scope boundary

**In scope:** the elliptic inverse problem as motivation; the discrete objective
$\Phi$, gradient, Hessian, normal equations; Elvetun–Nielsen weights as given;
DLRA, DLRA-CG, DLRA-PCG including restarts and preconditioners; comparison
against full-rank CG and baseline DLRA.

**Out of scope (do not pull in):**
- the randomized-SVD *construction* of $K$ (matrix-free rSVD, adjoint rSVD);
  $K$'s factors are given;
- the matrix-free / randomized *approximation of the weights*
  (`MatrixFreeENWeights`); $W$ is given;
- the transformed formulation as a solver (kept only as motivation for $W$).

Ground truth for validation: `ExactForwardOperator` (assembles $K$ densely for
small meshes). Reconstruction metrics available in `metrics.py` (Euclidean,
EMD/Wasserstein, IoU/AUC-IoU, SSIM).
