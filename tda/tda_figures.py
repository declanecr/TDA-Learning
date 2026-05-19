"""
tda.tda_figures — GridSpec figure orchestrators.

Imports computation from tda_data and renderers from tda_viz.
Each function owns its figure layout, calls plt.show(), and returns None.

Public API
----------
plot_noise_experiment(pts, rips, noise, thresholds, representations, mode)
plot_complex_comparison(result, threshold, representations)
plot_geometric(pts, alpha_value, shape, noise, circumsphere, circumcenter, seed)
plot_alpha_vr_comparison(pts, alpha_value, shape, noise, circumcenter, seed)
plot_distance_comparison(result)
plot_rips_comparison(result_a, result_b, label_a, label_b)
plot_alpha_comparison(result_a, result_b, label_a, label_b, alpha_value_a, alpha_value_b)
plot_full_analysis(result, threshold, alpha_value, circumcenter, seed)
print_distance_table(distances)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .tda_data import (
    BothResult, RipsResult,
    filter_diagrams, auto_alpha_value, diagram_distances,
)
from .tda_viz import (
    render_point_cloud, render_persistence_diagram,
    render_barcode, render_landscape,
    render_comparison_table, render_matching,
    render_voronoi_delaunay, render_alpha_overlay, render_vr_overlay,
    _ax_limits,
)


# ── Stage 01: noise / threshold sweep ────────────────────────────────────────

def plot_noise_experiment(
    pts: np.ndarray,
    rips: RipsResult,
    noise: float,
    thresholds,
    representations: tuple | list = ('pd', 'bc', 'pl'),
    mode: str = 'h1only',
) -> None:
    """
    Multi-panel Rips-only experiment figure.

    Parameters
    ----------
    pts             : point cloud (2D)
    rips            : RipsResult from compute_rips
    noise           : noise level shown in the title
    thresholds      : single float → one row of diagrams
                      [unfiltered, filtered] → two comparison rows
    representations : ordered subset of ('pd', 'bc', 'pl') — columns to include
    mode            : 'h1only'  show H₁ only in barcode / landscape columns
                      'overlay' show H₀+H₁ together
                      'grid'    2×2 sub-grid: H₁ | H₀+H₁  ×  unfiltered | filtered
                                (with a single threshold, 2×1: H₁ above / H₀+H₁ below)
    """
    _VALID_MODES = {'h1only', 'overlay', 'grid'}
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}; got {mode!r}")

    if not isinstance(thresholds, (list, tuple)):
        thresholds = [thresholds]
    single = len(thresholds) == 1

    rep_list = list(representations)
    n_rep = len(rep_list)
    total_cols = 1 + n_rep
    n_rows = 1 if single else 2

    cloud_ratio = max(1.0, 4 / total_cols)
    fig = plt.figure(figsize=(5 + 4 * n_rep, 4 if single else 7))
    gs = GridSpec(
        n_rows, total_cols, figure=fig,
        width_ratios=[cloud_ratio] + [1] * n_rep,
        hspace=0.08, wspace=0.4,
        left=0.05, right=0.97, top=0.88, bottom=0.10,
    )
    fig.suptitle(f"n={len(pts)}  |  noise={noise}", fontsize=12,
                 fontweight='bold', x=0.05, ha='left')

    # Point cloud spans all rows
    ax_cloud = fig.add_subplot(gs[:, 0])
    ax_cloud.scatter(pts[:, 0], pts[:, 1], s=8, color='steelblue', alpha=0.7)
    ax_cloud.set_aspect('equal')
    ax_cloud.set_title('Point cloud', fontsize=10)
    ax_cloud.tick_params(labelsize=8)

    filtered_sets = [filter_diagrams(rips.dgms, t) for t in thresholds]

    for col_offset, rep in enumerate(rep_list):
        col_idx = 1 + col_offset
        draw_fn = render_barcode if rep == 'bc' else render_landscape

        # ── Persistence diagram: one ax per threshold row ──────────────────
        if rep == 'pd':
            for row, (thresh, filtered) in enumerate(zip(thresholds, filtered_sets)):
                ax = fig.add_subplot(gs[row, col_idx])
                render_persistence_diagram(
                    filtered, ax,
                    title='Persistence Diagram' if row == 0 else f'PD  thresh={thresh}',
                )
                if single:
                    break

        # ── Barcode / Landscape: layout varies by mode ─────────────────────
        elif mode == 'grid' and single:
            gs_inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, col_idx], hspace=0.55)
            draw_fn(filtered_sets[0], fig.add_subplot(gs_inner[0]),
                    title='H₁', dims=(1,))
            draw_fn(filtered_sets[0], fig.add_subplot(gs_inner[1]),
                    title='H₀+H₁', dims=(0, 1))

        elif mode == 'grid' and not single:
            gs_inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, col_idx],
                                               hspace=0.55, wspace=0.25)
            for r, dims in enumerate([(1,), (0, 1)]):
                for c, (thresh, filtered) in enumerate(zip(thresholds, filtered_sets)):
                    h_label = 'H₁' if r == 0 else 'H₀+H₁'
                    t_label = 'unfilt' if c == 0 else f'th={thresh}'
                    draw_fn(filtered, fig.add_subplot(gs_inner[r, c]),
                            title=f'{h_label} {t_label}', dims=dims)

        elif mode == 'overlay' and single:
            draw_fn(filtered_sets[0], fig.add_subplot(gs[:, col_idx]),
                    title=f'H₀+H₁  th={thresholds[0]}', dims=(0, 1))

        elif mode == 'overlay' and not single:
            gs_inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, col_idx], hspace=0.55)
            for row, (thresh, filtered) in enumerate(zip(thresholds, filtered_sets)):
                label = 'unfiltered' if row == 0 else f'th={thresh}'
                draw_fn(filtered, fig.add_subplot(gs_inner[row]),
                        title=f'H₀+H₁ {label}', dims=(0, 1))

        elif single:  # h1only + single threshold
            draw_fn(filtered_sets[0], fig.add_subplot(gs[:, col_idx]),
                    title=f'H₁  th={thresholds[0]}', dims=(1,))

        else:  # h1only + two thresholds
            gs_inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, col_idx], hspace=0.55)
            for row, (thresh, filtered) in enumerate(zip(thresholds, filtered_sets)):
                label = 'unfiltered' if row == 0 else f'th={thresh}'
                draw_fn(filtered, fig.add_subplot(gs_inner[row]),
                        title=f'H₁ {label}', dims=(1,))

    plt.show()


# ── Stage 02: Rips vs Alpha comparison ───────────────────────────────────────

def plot_complex_comparison(
    result: BothResult,
    threshold: float = 0.0,
    representations: tuple | list = ('pd', 'bc', 'pl'),
) -> None:
    """
    2-row (Rips top, Alpha bottom) comparison figure.

    Parameters
    ----------
    result          : BothResult from compute_both
    threshold       : persistence threshold applied to both Rips and Alpha diagrams
    representations : ordered subset of ('pd', 'bc', 'pl') — columns to include
    """
    rips_f  = filter_diagrams(result.rips.dgms,  threshold)
    alpha_f = filter_diagrams(result.alpha.dgms, threshold)
    rep_list = list(representations)
    n_rep = len(rep_list)

    fig = plt.figure(figsize=(6 + 5 * n_rep, 9), layout='constrained')
    fig.suptitle(
        f"{result.shape.capitalize()}  |  n={result.n_points}"
        f"  |  noise={result.noise}  |  thresh={threshold}",
        fontsize=12, fontweight='bold',
    )
    gs = fig.add_gridspec(2, 1 + n_rep)

    ax_pc = fig.add_subplot(
        gs[:, 0], projection='3d' if result.pts.shape[1] == 3 else None
    )
    render_point_cloud(result.pts, ax_pc)

    for col_offset, rep in enumerate(rep_list):
        c = 1 + col_offset
        ax_r = fig.add_subplot(gs[0, c])
        ax_a = fig.add_subplot(gs[1, c])
        if rep == 'pd':
            render_persistence_diagram(
                rips_f, ax_r,
                title=f"Rips PD  |  {result.rips.num_edges} edges  {result.rips.elapsed_ms:.0f}ms",
            )
            render_persistence_diagram(
                alpha_f, ax_a,
                title=f"Alpha PD  |  {result.alpha.num_simplices} simplices  {result.alpha.elapsed_ms:.0f}ms",
            )
        elif rep == 'bc':
            render_barcode(rips_f,  ax_r, title="Rips Barcode")
            render_barcode(alpha_f, ax_a, title="Alpha Barcode")
        elif rep == 'pl':
            render_landscape(rips_f,  ax_r, title="Rips Landscape")
            render_landscape(alpha_f, ax_a, title="Alpha Landscape")

    plt.show()


# ── Stage 02: geometric visualization ────────────────────────────────────────

def plot_geometric(
    pts: np.ndarray,
    alpha_value: float,
    shape: str = '',
    noise: float | None = None,
    circumsphere: bool = False,
    circumcenter: bool = False,
    seed: int = 0,
) -> None:
    """
    Two-panel figure: Voronoi + Delaunay (left) | Alpha complex overlay (right). 2D only.

    alpha_value : GUDHI units — α = r² (squared circumradius).
    circumsphere: draw each Delaunay triangle's circumscribed circle in the left panel.
    circumcenter: draw circumcenter dots (green=inside, red=outside) in the right panel.
    """
    if pts.shape[1] != 2:
        raise ValueError("plot_geometric only supports 2D point clouds.")
    r = alpha_value ** 0.5
    noise_str = f"  |  noise={noise}" if noise is not None else ""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    fig.suptitle(
        f"{shape.capitalize()}  |  n={len(pts)}{noise_str}"
        f"  |  α={alpha_value} (r²)  →  r≈{r:.3f}",
        fontsize=12, fontweight='bold',
    )
    render_voronoi_delaunay(pts, ax1, circumsphere=circumsphere, seed=seed)
    render_alpha_overlay(pts, alpha_value, ax2, circumcenter=circumcenter, seed=seed)
    plt.show()


def plot_alpha_vr_comparison(
    pts: np.ndarray,
    alpha_value: float,
    shape: str = '',
    noise: float | None = None,
    circumcenter: bool = False,
    seed: int = 0,
) -> None:
    """
    Two-panel figure: Alpha complex (left) vs Vietoris-Rips (right) at the same ball radius.

    Both panels use r = sqrt(alpha_value). Alpha restricts simplices to pairs whose
    Voronoi cells intersect; VR includes every pair within 2r — so Alpha ⊆ VR always.

    alpha_value : GUDHI units — α = r² (squared circumradius).
    """
    r = alpha_value ** 0.5
    xlim, ylim = _ax_limits(pts)
    noise_str = f"  |  noise={noise}" if noise is not None else ""

    fig, (ax_a, ax_v) = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    fig.suptitle(
        f"{shape.capitalize()}  |  n={len(pts)}{noise_str}"
        f"  |  α={alpha_value:.4f}  →  r≈{r:.3f}",
        fontsize=12, fontweight='bold',
    )
    render_alpha_overlay(pts, alpha_value, ax_a, circumcenter=circumcenter, seed=seed)
    ax_a.set_xlim(xlim)
    ax_a.set_ylim(ylim)

    render_vr_overlay(pts, r, ax_v)
    ax_v.set_xlim(xlim)
    ax_v.set_ylim(ylim)

    plt.show()


# ── Stage 02: diagram distance comparison ────────────────────────────────────

def print_distance_table(distances: dict) -> None:
    """Print a Rips vs Alpha bottleneck / Wasserstein distance table to stdout."""
    col_w = 14
    header = f"{'dim':<6} {'bottleneck':>{col_w}} {'wasserstein':>{col_w}}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for dim, d in distances.items():
        print(f"H{dim}     {d['bottleneck']:>{col_w}.6f} {d['wasserstein']:>{col_w}.6f}")
    print(sep)


def plot_distance_comparison(result: BothResult) -> None:
    """
    2×2 figure: Rips PD | Alpha PD | H1 Bottleneck matching | H1 Wasserstein matching.
    Also prints the distance table to stdout.
    """
    distances = diagram_distances(result.rips.dgms, result.alpha.dgms)
    label = f"{result.shape.capitalize()}  |  noise={result.noise}"

    print(f"\nRips vs Alpha — {label}")
    print_distance_table(distances)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    ax_r, ax_a, ax_bn, ax_ws = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    render_persistence_diagram(result.rips.dgms,  ax_r, title=f"Rips — {label}")
    render_persistence_diagram(result.alpha.dgms, ax_a, title=f"Alpha — {label}")

    d1 = distances.get(1, {})
    d_r_fin = d1.get('d_rips',  np.empty((0, 2)))
    d_a_fin = d1.get('d_alpha', np.empty((0, 2)))
    have_h1 = len(d_r_fin) > 0 or len(d_a_fin) > 0

    if have_h1:
        render_matching(d_r_fin, d_a_fin, d1['bn_match'], ax_bn,
                        title=f"H1 Bottleneck  (dist={d1['bottleneck']:.4f})")
        render_matching(d_r_fin, d_a_fin, d1['ws_match'], ax_ws,
                        title=f"H1 Wasserstein  (dist={d1['wasserstein']:.4f})")
    else:
        for ax in (ax_bn, ax_ws):
            ax.text(0.5, 0.5, "No H1 points to match",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.axis('off')

    dist_lines = [
        f"H{dim}: bn={d['bottleneck']:.4f}  ws={d['wasserstein']:.4f}"
        for dim, d in distances.items()
    ]
    fig.suptitle("  ·  ".join(dist_lines), fontsize=10)
    plt.show()


def plot_rips_comparison(
    result_a: BothResult,
    result_b: BothResult,
    label_a: str = "Cloud A",
    label_b: str = "Cloud B",
) -> None:
    """
    3×2 figure comparing Rips diagrams from two point clouds:
      Row 0: point cloud A      | point cloud B
      Row 1: Rips PD A          | Rips PD B
      Row 2: H1 Bottleneck      | H1 Wasserstein
    Also prints the distance table to stdout.
    """
    distances = diagram_distances(result_a.rips.dgms, result_b.rips.dgms)

    print(f"\nRips vs Rips — {label_a}  vs  {label_b}")
    print_distance_table(distances)

    is_3d = result_a.pts.shape[1] == 3
    fig = plt.figure(figsize=(14, 14), layout='constrained')
    gs = fig.add_gridspec(3, 2)

    ax_ca = fig.add_subplot(gs[0, 0], projection='3d' if is_3d else None)
    ax_cb = fig.add_subplot(gs[0, 1], projection='3d' if is_3d else None)
    ax_pa = fig.add_subplot(gs[1, 0])
    ax_pb = fig.add_subplot(gs[1, 1])
    ax_bn = fig.add_subplot(gs[2, 0])
    ax_ws = fig.add_subplot(gs[2, 1])

    render_point_cloud(result_a.pts, ax_ca, title=label_a)
    render_point_cloud(result_b.pts, ax_cb, title=label_b)
    render_persistence_diagram(result_a.rips.dgms, ax_pa, title=f"Rips PD — {label_a}")
    render_persistence_diagram(result_b.rips.dgms, ax_pb, title=f"Rips PD — {label_b}")

    d1 = distances.get(1, {})
    d_a_fin = d1.get('d_rips',  np.empty((0, 2)))
    d_b_fin = d1.get('d_alpha', np.empty((0, 2)))
    have_h1 = len(d_a_fin) > 0 or len(d_b_fin) > 0

    if have_h1:
        render_matching(d_a_fin, d_b_fin, d1['bn_match'], ax_bn,
                        title=f"H1 Bottleneck  (dist={d1['bottleneck']:.4f})",
                        label_a=label_a, label_b=label_b)
        render_matching(d_a_fin, d_b_fin, d1['ws_match'], ax_ws,
                        title=f"H1 Wasserstein  (dist={d1['wasserstein']:.4f})",
                        label_a=label_a, label_b=label_b)
    else:
        for ax in (ax_bn, ax_ws):
            ax.text(0.5, 0.5, "No H1 points to match",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.axis('off')

    dist_lines = [
        f"H{dim}: bn={d['bottleneck']:.4f}  ws={d['wasserstein']:.4f}"
        for dim, d in distances.items()
    ]
    fig.suptitle(
        f"Rips vs Rips  ·  {label_a}  vs  {label_b}\n" + "  ·  ".join(dist_lines),
        fontsize=10,
    )
    plt.show()


def plot_alpha_comparison(
    result_a: BothResult,
    result_b: BothResult,
    label_a: str = "Cloud A",
    label_b: str = "Cloud B",
    alpha_value_a: float | None = None,
    alpha_value_b: float | None = None,
) -> None:
    """
    3×2 figure comparing Alpha diagrams from two point clouds:
      Row 0: Alpha overlay A (2D) or point cloud A (3D) | same for B
      Row 1: Alpha PD A                                 | Alpha PD B
      Row 2: H1 Bottleneck                              | H1 Wasserstein
    Also prints the distance table to stdout.

    alpha_value_a/b : GUDHI r² units. Auto-derived from the top-persistence
                      H1 bar if None.
    """
    if alpha_value_a is None:
        alpha_value_a = auto_alpha_value(result_a.alpha.dgms)
    if alpha_value_b is None:
        alpha_value_b = auto_alpha_value(result_b.alpha.dgms)

    distances = diagram_distances(result_a.alpha.dgms, result_b.alpha.dgms)

    print(f"\nAlpha vs Alpha — {label_a}  vs  {label_b}")
    print_distance_table(distances)

    is_3d = result_a.pts.shape[1] == 3
    fig = plt.figure(figsize=(14, 14), layout='constrained')
    gs = fig.add_gridspec(3, 2)

    ax_ca = fig.add_subplot(gs[0, 0], projection='3d' if is_3d else None)
    ax_cb = fig.add_subplot(gs[0, 1], projection='3d' if is_3d else None)
    ax_pa = fig.add_subplot(gs[1, 0])
    ax_pb = fig.add_subplot(gs[1, 1])
    ax_bn = fig.add_subplot(gs[2, 0])
    ax_ws = fig.add_subplot(gs[2, 1])

    if is_3d:
        render_point_cloud(result_a.pts, ax_ca, title=label_a)
        render_point_cloud(result_b.pts, ax_cb, title=label_b)
    else:
        render_alpha_overlay(result_a.pts, alpha_value_a, ax_ca)
        ax_ca.set_title(f"{label_a}  (α={alpha_value_a:.4f})")
        render_alpha_overlay(result_b.pts, alpha_value_b, ax_cb)
        ax_cb.set_title(f"{label_b}  (α={alpha_value_b:.4f})")

    render_persistence_diagram(result_a.alpha.dgms, ax_pa, title=f"Alpha PD — {label_a}")
    render_persistence_diagram(result_b.alpha.dgms, ax_pb, title=f"Alpha PD — {label_b}")

    d1 = distances.get(1, {})
    d_a_fin = d1.get('d_rips',  np.empty((0, 2)))
    d_b_fin = d1.get('d_alpha', np.empty((0, 2)))
    have_h1 = len(d_a_fin) > 0 or len(d_b_fin) > 0

    if have_h1:
        render_matching(d_a_fin, d_b_fin, d1['bn_match'], ax_bn,
                        title=f"H1 Bottleneck  (dist={d1['bottleneck']:.4f})",
                        label_a=label_a, label_b=label_b)
        render_matching(d_a_fin, d_b_fin, d1['ws_match'], ax_ws,
                        title=f"H1 Wasserstein  (dist={d1['wasserstein']:.4f})",
                        label_a=label_a, label_b=label_b)
    else:
        for ax in (ax_bn, ax_ws):
            ax.text(0.5, 0.5, "No H1 points to match",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.axis('off')

    dist_lines = [
        f"H{dim}: bn={d['bottleneck']:.4f}  ws={d['wasserstein']:.4f}"
        for dim, d in distances.items()
    ]
    fig.suptitle(
        f"Alpha vs Alpha  ·  {label_a}  vs  {label_b}\n" + "  ·  ".join(dist_lines),
        fontsize=10,
    )
    plt.show()


# ── Stage 02: unified full analysis ──────────────────────────────────────────

def plot_full_analysis(
    result: BothResult,
    threshold: float = 0.0,
    alpha_value: float | None = None,
    circumcenter: bool = False,
    seed: int = 0,
) -> None:
    """
    Unified 5-panel figure for one dataset:

      [0, 0]  Point cloud          [0, 1]  Rips PD        [0, 2]  Alpha PD
      [1:, 0] Alpha complex overlay [1, 1]  Rips Barcode   [1, 2]  Alpha Barcode
                                    [2, 1:] Summary table

    alpha_value : GUDHI r² units. Auto-derived from the top-persistence H1 bar if None.
    """
    rips_f  = filter_diagrams(result.rips.dgms,  threshold)
    alpha_f = filter_diagrams(result.alpha.dgms, threshold)
    if alpha_value is None:
        alpha_value = auto_alpha_value(result.alpha.dgms)
    is_3d = result.pts.shape[1] == 3

    fig = plt.figure(figsize=(16, 10), layout='constrained')
    fig.suptitle(
        f"{result.shape.capitalize()}  |  n={result.n_points}"
        f"  |  noise={result.noise}  |  thresh={threshold}  |  α={alpha_value:.4f}",
        fontsize=12, fontweight='bold',
    )
    gs = fig.add_gridspec(3, 3)

    ax_pc    = fig.add_subplot(gs[0, 0], projection='3d' if is_3d else None)
    ax_alpha = fig.add_subplot(gs[1:3, 0])
    ax_pd_r  = fig.add_subplot(gs[0, 1])
    ax_pd_a  = fig.add_subplot(gs[0, 2])
    ax_bc_r  = fig.add_subplot(gs[1, 1])
    ax_bc_a  = fig.add_subplot(gs[1, 2])
    ax_table = fig.add_subplot(gs[2, 1:3])

    render_point_cloud(result.pts, ax_pc)

    if not is_3d:
        render_alpha_overlay(result.pts, alpha_value, ax_alpha,
                             circumcenter=circumcenter, seed=seed)
    else:
        ax_alpha.text(0.5, 0.5, "Alpha overlay\n(2D only)",
                      ha='center', va='center', transform=ax_alpha.transAxes,
                      fontsize=12, color='gray')
        ax_alpha.axis('off')

    render_persistence_diagram(
        rips_f,  ax_pd_r, title=f"Rips PD  |  {result.rips.elapsed_ms:.0f}ms")
    render_persistence_diagram(
        alpha_f, ax_pd_a, title=f"Alpha PD  |  {result.alpha.elapsed_ms:.0f}ms")
    render_barcode(rips_f,  ax_bc_r, title="Rips Barcode")
    render_barcode(alpha_f, ax_bc_a, title="Alpha Barcode")
    render_comparison_table(result.rips, result.alpha, ax_table)

    plt.show()
