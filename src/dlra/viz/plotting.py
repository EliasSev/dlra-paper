"""
Plotting helpers. Palette and style constants live in `dlra.viz.colors`.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from dlra.viz.colors import C5, C3, C2, CMAP, M, MS, LS  # re-exported for convenience


def save_plot(fig_name: str) -> None:
    """Save the current figure to ../../figures/<fig_name>.png if it does not exist."""
    if fig_name is None:
        return
    path = Path(f"../../figures/{fig_name}.png")
    if not path.exists():
        plt.savefig(path, dpi=300, bbox_inches="tight")
    else:
        print(path, "already exists")