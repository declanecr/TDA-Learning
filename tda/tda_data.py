"""
tda.tda_data — pure computation layer, no matplotlib.

Public API
----------
generate_point_cloud(shape, n_points, noise) -> ndarray   # 'circle'|'torus'|'sphere'|'figure8'
compute_rips(pts, max_dim)                   -> RipsResult
compute_alpha(pts, max_dim)                  -> AlphaResult
compute_both(shape, n_points, noise, max_dim)-> BothResult
compute_cubical(img, max_dim)                -> CubicalResult
compute_mnist(digit, instance, max_dim)      -> MNISTResult
filter_diagrams(dgms, threshold)             -> list[ndarray]
h1_stats(dgms)                               -> (int, float)
auto_alpha_value(dgms_alpha)                 -> float
diagram_distances(dgms_rips, dgms_alpha)     -> dict
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tadasets
import gudhi
from ripser import ripser
from persim import bottleneck, wasserstein


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RipsResult:
    dgms: list
    num_edges: int
    elapsed_ms: float


@dataclass
class AlphaResult:
    dgms: list
    num_simplices: int
    elapsed_ms: float


@dataclass
class BothResult:
    pts: np.ndarray
    shape: str
    n_points: int
    noise: float
    rips: RipsResult
    alpha: AlphaResult


@dataclass
class CubicalResult:
    dgms: list
    num_simplices: int
    elapsed_ms: float


@dataclass
class MNISTResult:
    digit: int
    instance: int
    img: np.ndarray     # shape (28, 28), float64, values 0–255
    cubical: CubicalResult


# ── Point cloud generation ────────────────────────────────────────────────────

def generate_point_cloud(shape: str, n_points: int, noise: float) -> np.ndarray:
    """
    Return a noisy point cloud sampled from the given shape.

    shape    : 'circle' | 'torus' | 'sphere' | 'figure8'
    n_points : number of sample points
    """
    if shape == 'circle':
        return tadasets.dsphere(n=n_points, d=1, r=1, noise=noise)
    if shape == 'torus':
        return tadasets.torus(n=n_points, c=2, a=1, noise=noise)
    if shape == 'sphere':
        return tadasets.dsphere(n=n_points, d=2, r=1, noise=noise)
    if shape == 'figure8':
        return tadasets.infty_sign(n=n_points, noise=noise)
    raise ValueError(
        f"Unknown shape {shape!r}. Choose: 'circle', 'torus', 'sphere', 'figure8'."
    )


# ── Persistence computation ───────────────────────────────────────────────────

def compute_rips(pts: np.ndarray, max_dim: int = 2) -> RipsResult:
    """Run Ripser on pts and return a RipsResult."""
    t0 = time.time()
    result = ripser(pts, maxdim=max_dim)
    return RipsResult(
        dgms=result['dgms'],
        num_edges=result['num_edges'],
        elapsed_ms=(time.time() - t0) * 1000,
    )


def compute_alpha(pts: np.ndarray, max_dim: int = 2) -> AlphaResult:
    """
    Run GUDHI AlphaComplex on pts and return an AlphaResult.

    Diagram values are in GUDHI units: α = r² (squared circumradius).
    Essential classes (death=inf) are preserved.
    """
    t0 = time.time()
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()
    st.compute_persistence()
    dgms = [
        np.array(
            [[b, d] for dim, (b, d) in st.persistence() if dim == i]
            or np.empty((0, 2))
        )
        for i in range(max_dim + 1)
    ]
    return AlphaResult(
        dgms=dgms,
        num_simplices=st.num_simplices(),
        elapsed_ms=(time.time() - t0) * 1000,
    )


def compute_both(shape: str, n_points: int, noise: float,
                 max_dim: int = 2) -> BothResult:
    """
    Generate a point cloud and compute both Rips and Alpha persistence.
    Convenience wrapper around generate_point_cloud + compute_rips + compute_alpha.
    """
    pts = generate_point_cloud(shape, n_points, noise)
    return BothResult(
        pts=pts,
        shape=shape,
        n_points=n_points,
        noise=noise,
        rips=compute_rips(pts, max_dim),
        alpha=compute_alpha(pts, max_dim),
    )


# ── MNIST / cubical persistence ──────────────────────────────────────────────

def _load_mnist_image(digit: int, instance: int) -> np.ndarray:
    """Return the (28, 28) grayscale image for the given digit class and instance index."""
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    idx = np.where(mnist.target == str(digit))[0][instance]
    return mnist.data[idx].reshape(28, 28).astype(float)  # values 0.0–255.0


def compute_cubical(img: np.ndarray, max_dim: int = 1) -> CubicalResult:
    """
    Run GUDHI CubicalComplex on a grayscale image using a sublevel filtration
    on the inverted normalised pixel values (ink=0 enters first, background=1 last).

    The image is normalised to [0, 1] then inverted, so ink pixels enter the
    filtration first. H1 bars correspond to loops within the ink strokes.

    img     : (H, W) ndarray, any non-negative scale
    max_dim : maximum homology dimension (default 1; H2 is trivial for 2D grids)
    """
    t0 = time.time()
    h, w = img.shape
    scale = img.max() if img.max() > 0 else 1.0
    cc = gudhi.CubicalComplex(
        dimensions=[h, w],
        top_dimensional_cells=(1 - img / scale).flatten(),
    )
    cc.compute_persistence()
    pairs = cc.persistence()
    dgms = [
        np.array([[b, d] for dim, (b, d) in pairs if dim == i])
        if any(dim == i for dim, _ in pairs)
        else np.empty((0, 2))
        for i in range(max_dim + 1)
    ]
    return CubicalResult(
        dgms=dgms,
        num_simplices=cc.num_simplices(),
        elapsed_ms=(time.time() - t0) * 1000,
    )


def compute_mnist(digit: int, instance: int = 0,
                  max_dim: int = 1) -> MNISTResult:
    """
    Load an sklearn digit image and compute cubical persistence on it.

    digit    : class label 0–9
    instance : which occurrence of that digit to use (default 0)
    max_dim  : passed to compute_cubical (default 1)
    """
    img = _load_mnist_image(digit, instance)
    cubical = compute_cubical(img, max_dim=max_dim)
    return MNISTResult(digit=digit, instance=instance, img=img, cubical=cubical)


# ── Diagram utilities ─────────────────────────────────────────────────────────

def filter_diagrams(dgms: list, threshold: float) -> list:
    """
    Drop finite bars with persistence ≤ threshold; always keep infinite bars.

    Works on a list of diagram arrays (one per homology dimension).
    Pass threshold=0 to skip filtering entirely.
    """
    if threshold <= 0:
        return dgms
    out = []
    for dgm in dgms:
        if len(dgm) == 0:
            out.append(dgm)
            continue
        inf_mask = ~np.isfinite(dgm[:, 1])
        fin_mask = np.isfinite(dgm[:, 1]) & ((dgm[:, 1] - dgm[:, 0]) > threshold)
        out.append(dgm[inf_mask | fin_mask])
    return out


def h1_stats(dgms: list) -> tuple[int, float]:
    """Return (finite H1 bar count, max H1 persistence) for a diagram list."""
    if len(dgms) < 2 or len(dgms[1]) == 0:
        return 0, float('nan')
    h1 = dgms[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if len(finite) == 0:
        return 0, float('nan')
    persistences = finite[:, 1] - finite[:, 0]
    return len(finite), float(persistences.max())


def auto_alpha_value(dgms_alpha: list) -> float:
    """
    Return the birth value of the top-persistence finite H1 bar (GUDHI r² units).
    Falls back to 0.05 if no finite H1 bars exist.
    """
    if len(dgms_alpha) < 2 or len(dgms_alpha[1]) == 0:
        return 0.05
    h1 = dgms_alpha[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if len(finite) == 0:
        return 0.05
    idx = np.argmax(finite[:, 1] - finite[:, 0])
    return max(float(finite[idx, 0]), 1e-3)


def diagram_distances(dgms_rips: list, dgms_alpha: list) -> dict:
    """
    Compute bottleneck and Wasserstein-1 distances between Rips and Alpha diagrams
    for each shared homology dimension.

    Returns a dict keyed by dimension:
        {dim: {'bottleneck': float, 'wasserstein': float,
               'bn_match': ndarray, 'ws_match': ndarray,
               'd_rips': ndarray, 'd_alpha': ndarray}}

    'd_rips' and 'd_alpha' are the finite-bar subsets used for matching.
    """
    n_dims = min(len(dgms_rips), len(dgms_alpha))
    results = {}
    for dim in range(n_dims):
        d_r = dgms_rips[dim]
        d_a = dgms_alpha[dim]
        d_r_fin = d_r[np.isfinite(d_r[:, 1])] if len(d_r) else d_r
        d_a_fin = d_a[np.isfinite(d_a[:, 1])] if len(d_a) else d_a

        if len(d_r_fin) == 0 and len(d_a_fin) == 0:
            results[dim] = dict(
                bottleneck=0.0, wasserstein=0.0,
                bn_match=np.empty((0, 2)), ws_match=np.empty((0, 2)),
                d_rips=d_r_fin, d_alpha=d_a_fin,
            )
        else:
            bn, bn_match = bottleneck(d_r_fin, d_a_fin, matching=True)
            ws, ws_match = wasserstein(d_r_fin, d_a_fin, matching=True)
            results[dim] = dict(
                bottleneck=bn, wasserstein=ws,
                bn_match=bn_match, ws_match=ws_match,
                d_rips=d_r_fin, d_alpha=d_a_fin,
            )
    return results
