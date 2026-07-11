from dlra.problem import TestProblemSetup, TestProblem, ForwardOperator
from dlra.evaluation import error3, error_ssim, error_auc_iou
from dlra.viz import save_plot
from dlra.solvers import (
    ConjugateGradient,
    DynamicalLowRankApproximation, DynamicalLowRankCG, DynamicalLowRankPCG
)

# Note: `dlra.io` (progress_bar, disk_cache) and `dlra.rsvd` (out of scope; the
# rSVD construction of K) are intentionally not re-exported here -- import them
# from their submodules.

__all__ = [
    "TestProblemSetup", "TestProblem", "ForwardOperator",
    "ConjugateGradient",
    "DynamicalLowRankApproximation", "DynamicalLowRankCG", "DynamicalLowRankPCG",
    "error3", "error_ssim", "error_auc_iou",
    "save_plot",
]