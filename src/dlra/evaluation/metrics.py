"""
Reconstruction error metrics comparing a recovered source to the ground truth:
Euclidean distance, earth-mover / Wasserstein distance (EMD), thresholded IoU
(and its AUC over thresholds), SSIM, and centroid/correlation shifts.

Also holds the grid <-> FE-DOF indexing helpers (`SpaceIndexing`,
`matrix_to_vec`, `vec_to_matrix`) used to move between the n x n image layout and
the FE coefficient vector.
"""
import numpy as np
from fenics import FunctionSpace, Point
from skimage.segmentation import chan_vese
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim


def relative_segmentation(x: np.ndarray, tau: float) -> np.ndarray:
    """Threshold at a fraction `tau` of the max, giving a binary mask."""
    return x >= tau * np.max(x)


def error_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return intersection / union


def error_auc_iou(
        x: np.ndarray, x_hat: np.ndarray, tau_range: np.ndarray = np.linspace(0.1, 1, 100)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Area under the IoU-vs-threshold curve.

    x, np.array         : ground truth
    x_hat, np.array     : reconstruction
    tau_range, np.array : thresholds as fraction of max

    returns: (auc_iou, tau_max, ious)
    """
    ious = np.zeros(len(tau_range))
    for i, tau in enumerate(tau_range):
        mask = relative_segmentation(x, tau)
        mask_hat = relative_segmentation(x_hat, tau)
        ious[i] = error_iou(mask, mask_hat)

    return np.trapz(ious, tau_range), tau_range[np.argmax(ious)], ious


def error3(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """
    Compute the error triplet {'euclidean', 'emd', 'auc_iou'}.

    x, np.array     : ground truth
    x_hat, np.array : reconstruction
    """
    return {
        'euclidean': np.linalg.norm(x - x_hat),
        'emd': error_movers(x, x_hat),
        'auc_iou': error_auc_iou(x, x_hat)[0],
    }


class SpaceIndexing:
    """Indexing and dimension info about the function space V_h."""
    def __init__(self, V_h: FunctionSpace):
        self.coords = V_h.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((self.coords[:, 0], self.coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(V_h.dim()))


def matrix_to_vec(X, space: SpaceIndexing):
    """Image matrix -> FE coefficient vector (permutes by FE-DOF order)."""
    return X.flatten()[space.dof_indices]


def vec_to_matrix(x, space: SpaceIndexing):
    """FE coefficient vector -> n x n image matrix (permutes by FE-DOF order)."""
    return x[space.grid_indices].reshape((space.n, space.n))


def centroid(x):
    idx = np.arange(len(x))
    return np.sum(idx * x) / np.sum(x)


def error_centroid(x, x_hat):
    return abs(centroid(x) - centroid(x_hat))


def error_correlation(x, x_hat):
    corr = np.correlate(x, x_hat, mode='full')
    shift = np.argmax(corr) - (len(x) - 1)
    return shift


def error_movers(x, x_hat):
    """Earth-mover / Wasserstein distance between |x| and |x_hat|."""
    i = np.arange(len(x))
    return wasserstein_distance(i, i, np.abs(x), np.abs(x_hat))


def rectangular_interpolation(mesh, f):
    """Interpolate an FE function f onto a normalized square grid Z."""
    coords = mesh.coordinates()

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    num_nodes = mesh.num_vertices()
    nx = ny = int(np.sqrt(num_nodes))
    nx, ny = int(nx * 1.2), int(ny * 1.2)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    tree = mesh.bounding_box_tree()
    for j in range(ny):
        for i in range(nx):
            p = Point(X[j, i], Y[j, i])
            if tree.compute_first_entity_collision(p) < mesh.num_cells():
                Z[j, i] = f(p)
            else:
                Z[j, i] = np.nan  # outside domain

    return (Z - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z))


def compute_cv_mask(X, mu=0.1, lambda1=1, lambda2=1):
    """Chan-Vese segmentation mask of an image X."""
    if np.isnan(X).any():
        X = np.nan_to_num(X, copy=True, nan=0.0)
    return chan_vese(X, mu=mu, lambda1=lambda1, lambda2=lambda2)


def error_ssim(X, X_hat):
    """SSIM, taking the better of X_hat and its complement (sign-invariant)."""
    X = X.astype(float)
    X_hat = X_hat.astype(float)
    s1 = ssim(X, X_hat, data_range=1.0)
    s2 = ssim(X, 1 - X_hat, data_range=1.0)
    return max(s1, s2)