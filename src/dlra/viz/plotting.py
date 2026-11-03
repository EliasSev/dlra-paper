"""
Plotting helpers. Palette and style constants live in `dlra.viz.colors`.
"""
from pathlib import Path
from typing import Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from fenics import Function, FunctionSpace, plot

from dlra.viz.colors import C5, C3, C2, CMAP, M, MS, LS  # re-exported for convenience


def plot_source(
    sources: Union[NDArray, Sequence[NDArray]],
    Vh: FunctionSpace,
    titles: Union[str, Sequence[str], None] = None,
    cmap: str = CMAP,
    colorbar: bool = True,
    axes: Union[plt.Axes, Sequence[plt.Axes], None] = None,
):
    """
    Plot one source, or a list of sources side by side, via `fenics.plot`.

    sources:  a single coefficient vector (1D array), or a list of them.
    Vh:       the FE `FunctionSpace` the coefficient vectors live in; used to
              wrap each vector into a `Function` for plotting.
    titles:   optional title, or one title per source.
    cmap:     matplotlib colormap name (defaults to `dlra.viz.CMAP`).
    colorbar: draw a colorbar beside each field.
    axes:     existing `Axes` (or list) to draw into; created if None.

    Returns the `Axes` (single) or list of `Axes` used.
    """
    single = isinstance(sources, np.ndarray)
    vecs = [sources] if single else list(sources)

    fns = []
    for v in vecs:
        f = Function(Vh)
        f.vector()[:] = np.asarray(v)
        fns.append(f)

    if isinstance(titles, str) or titles is None:
        titles = [titles] * len(fns)

    if axes is None:
        fig, axs = plt.subplots(1, len(fns), figsize=(4 * len(fns), 4), squeeze=False)
        axs = list(axs.ravel())
    else:
        axs = [axes] if isinstance(axes, plt.Axes) else list(axes)

    for ax, f, title in zip(axs, fns, titles):
        plt.sca(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        mappable = plot(f, cmap=cmap)
        ax.set_aspect("equal")
        if title is not None:
            ax.set_title(title)
        if colorbar:
            ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    return axs[0] if single else axs


def save_plot(fig_name: str) -> None:
    """Save the current figure to ../../figures/<fig_name>.png if it does not exist."""
    if fig_name is None:
        return
    path = Path(f"../../figures/{fig_name}.png")
    if not path.exists():
        plt.savefig(path, dpi=300, bbox_inches="tight")
    else:
        print(path, "already exists")