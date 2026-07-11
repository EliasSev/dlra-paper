"""
Test-problem registry and builder for the elliptic source-identification problem.

A `TestProblem` bundles a discretized problem: the forward operator (with its FE
matrices and weights), the true source, and the boundary data y = K x. The
`TestProblemSetup` builder turns a named spec from `PROBLEM_REGISTRY` into one.
"""
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from fenics import Function, FunctionSpace

from dlra.problem.meshes import get_square_mesh, get_L_mesh
from dlra.problem.sources import get_square_f, get_Gaussian_f, get_disk_f
from dlra.problem.fem import ForwardOperator

# Source 'kind' -> the mesh_utils constructor that builds it. The remaining
# entries of a source spec are forwarded as keyword arguments.
SOURCE_BUILDERS: dict[str, Callable[..., Function]] = {
    "square":   get_square_f,
    "gaussian": get_Gaussian_f,
    "disk":     get_disk_f,
}


@dataclass(frozen=True)
class ProblemSpec:
    """Declarative description of a test problem."""
    mesh_factory: Callable[[int], Any]
    sources: list[dict[str, Any]]  # each: {"kind": ..., **builder_kwargs}


@dataclass(frozen=True)
class TestProblem:
    """A fully set-up inverse problem."""
    name: str
    operator: ForwardOperator      # FE matrices, K, weights
    f: Function                    # true source (FE function)
    x: NDArray                     # true source coefficients
    y: NDArray                     # boundary data, y = K x (clean)
    y_noisy: NDArray               # y + noise (== y when noise_level=0)
    spec: ProblemSpec

    @property
    def Vh(self) -> FunctionSpace:
        return self.operator.Vh

    @property
    def N(self) -> int:
        return self.operator.N

    @property
    def N_b(self) -> int:
        return self.operator.N_b

    @property
    def M_dx(self) -> csr_matrix:
        return self.operator.M_dx

    @property
    def M_ds(self) -> csr_matrix:
        return self.operator.M_ds

    @property
    def K(self) -> NDArray:
        return self.operator.K

    @property
    def weights(self) -> NDArray:
        return self.operator.weights


PROBLEM_REGISTRY: dict[str, ProblemSpec] = {
    "I": ProblemSpec(
        mesh_factory=get_square_mesh,
        sources=[{"kind": "square", "x0": 0.20, "y0": 0.20, "w": 0.15, "h": 0.15}],
    ),
    "II": ProblemSpec(
        mesh_factory=get_square_mesh,
        sources=[
            {"kind": "square", "x0": 0.10, "y0": 0.10, "w": 0.15, "h": 0.15},
            {"kind": "square", "x0": 0.75, "y0": 0.75, "w": 0.15, "h": 0.15},
            {"kind": "square", "x0": 0.15, "y0": 0.70, "w": 0.15, "h": 0.15},
        ],
    ),
    "III": ProblemSpec(
        mesh_factory=get_L_mesh,
        sources=[
            {"kind": "square", "x0": 0.20, "y0": 0.20, "w": 0.25, "h": 0.25},
            {"kind": "square", "x0": 1.55, "y0": 0.55, "w": 0.25, "h": 0.25},
        ],
    ),
}


class TestProblemSetup:
    """
    Builds `TestProblem`s from `PROBLEM_REGISTRY`.

    n:           Mesh resolution (~ n x n nodes).
    sigma, c:    PDE coefficients of -div(sigma grad u) + c u = f.
    noise_level: Relative Gaussian noise on y (0.0 = clean). See `_add_noise`.
    seed:        RNG seed for reproducible noise.
    """

    def __init__(
        self,
        n: int,
        sigma: float = 1.0,
        c: float = 1.0,
        noise_level: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.sigma = sigma
        self.c = c
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def build(self, name: str) -> TestProblem:
        """Set up a single named problem."""
        spec = PROBLEM_REGISTRY[name]
        Vh = FunctionSpace(spec.mesh_factory(self.n), "CG", 1)

        operator = ForwardOperator(Vh, sigma=self.sigma, c=self.c)
        f, x = self._build_source(Vh, spec.sources)
        y = operator.apply_K(x)
        y_noisy = self._add_noise(y)

        return TestProblem(
            name=name,
            operator=operator,
            f=f,
            x=x,
            y=y,
            y_noisy=y_noisy,
            spec=spec
        )

    def build_all(self) -> dict[str, TestProblem]:
        """Set up every problem in the registry."""
        return {name: self.build(name) for name in PROBLEM_REGISTRY}

    def _build_source(
        self, Vh: FunctionSpace, sources: list[dict[str, Any]]
    ) -> tuple[Function, NDArray]:
        """Sum the listed source terms into a single source function and vector."""
        x = np.zeros(Vh.dim())
        for s in sources:
            kwargs = {k: v for k, v in s.items() if k != "kind"}
            x += SOURCE_BUILDERS[s["kind"]](Vh, **kwargs).vector().get_local()

        f = Function(Vh)
        f.vector()[:] = x
        return f, x

    def _add_noise(self, y: NDArray) -> NDArray:
        """Additive Gaussian noise scaled to `noise_level * ||y|| / sqrt(N_b)`."""
        if self.noise_level <= 0.0:
            return y.copy()
        scale = self.noise_level * np.linalg.norm(y) / np.sqrt(y.size)
        return y + scale * self.rng.standard_normal(y.shape)
