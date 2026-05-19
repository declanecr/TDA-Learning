"""
tda.tda_viz — atomic ax-based renderers.

No computation, no plt.show(), no GridSpec. Every function takes
pre-computed data and one or more axes and returns None.

Public API
----------
render_point_cloud(pts, ax, title)
render_persistence_diagram(dgms, ax, title)
render_barcode(dgms, ax, title, dims)
render_landscape(dgms, ax, title, n_landscapes, dims)
render_comparison_table(rips, alpha, ax)
render_matching(d_r, d_a, matching, ax, title, label_a, label_b)
render_voronoi_delaunay(pts, ax, circumsphere, seed)
render_alpha_overlay(pts, alpha_value, ax, circumcenter, seed)
render_vr_overlay(pts, radius, ax)
render_mnist_image(img, ax, title, digit)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import gudhi
from persim import plot_diagrams
from scipy.spatial import Delaunay, Voronoi
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch

from .tda_data import RipsResult, AlphaResult, h1_stats


# ── Private geometry helpers ──────────────────────────────────────────────────

def _circumcenter_r2(p0, p1, p2):
    """Circumcenter and squared circumradius of triangle (p0, p1, p2)."""
    a, b = p1 - p0, p2 - p0
    D = 2.0 * (a[0] * b[1] - a[1] * b[0])
    if abs(D) < 1e-12:
        return None, None
    ux = (b[1] * (a @ a) - a[1] * (b @ b)) / D
    uy = (a[0] * (b @ b) - b[0] * (a @ a)) / D
    cc = p0 + np.array([ux, uy])
    return cc, float((cc[0] - p0[0]) ** 2 + (cc[1] - p0[1]) ** 2)


def _voronoi_finite_polygons_2d(vor, radius=None):
    """Extend infinite Voronoi ridges to finite polygons sorted CCW."""
    new_regions, new_verts = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = (vor.points.max() - vor.points.min()) * 3

    all_ridges: dict = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        verts = vor.regions[region_idx]
        if all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue
        new_region = [v for v in verts if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_verts.append(far_point.tolist())
            new_region.append(len(new_verts) - 1)
        vs = np.array([new_verts[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_regions.append(np.array(new_region)[np.argsort(angles)].tolist())

    return new_regions, np.array(new_verts)


def _build_voronoi_colors(pts, seed=0):
    """Return (regions, verts, cell_colors) for pts using HSV palette."""
    vor = Voronoi(pts)
    regions, verts = _voronoi_finite_polygons_2d(vor)
    rng = np.random.default_rng(seed)
    hues = rng.permutation(np.linspace(0, 1, len(pts), endpoint=False))
    cell_colors = plt.cm.hsv(hues)
    cell_colors[:, :3] = cell_colors[:, :3] * 0.35 + 0.65
    return regions, verts, cell_colors


def _ax_limits(pts, pad_frac=0.05):
    """Return ((xmin, xmax), (ymin, ymax)) with fractional padding."""
    pad = (pts.max() - pts.min()) * pad_frac
    return (pts[:, 0].min() - pad, pts[:, 0].max() + pad), \
           (pts[:, 1].min() - pad, pts[:, 1].max() + pad)


# ── Public renderers ──────────────────────────────────────────────────────────

def render_point_cloud(pts: np.ndarray, ax, title: str = "Point Cloud") -> None:
    """Scatter a 2D or 3D point cloud onto ax."""
    if pts.shape[1] == 2:
        ax.scatter(pts[:, 0], pts[:, 1], s=5, color='steelblue', alpha=0.8)
        ax.set_aspect('equal')
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, color='steelblue', alpha=0.8)
    ax.set_title(title)


def render_persistence_diagram(dgms: list, ax,
                                title: str = "Persistence Diagram") -> None:
    """Render a persistence diagram using persim on ax."""
    plot_diagrams(dgms, show=False, ax=ax)
    ax.set_title(title)


def render_barcode(dgms: list, ax, title: str = "Barcode",
                   dims: tuple | None = None) -> None:
    """
    Render a barcode on ax. Arrow tips indicate infinite bars.

    dims : tuple of homology dimensions to include, e.g. (1,) or (0, 1).
           None renders all available dimensions.
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    active = list(range(len(dgms))) if dims is None else list(dims)
    all_finite = [d for i in active for _, d in dgms[i] if np.isfinite(d)]
    x_right = max(all_finite) if all_finite else 1.0
    inf_clip = x_right * 1.12
    y = 0
    for dim in active:
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        color = colors[dim % len(colors)]
        for b, d in dgm:
            end = d if np.isfinite(d) else inf_clip
            ax.plot([b, end], [y, y], color=color, linewidth=1.5, solid_capstyle='butt')
            if not np.isfinite(d):
                ax.plot(end, y, '>', color=color, markersize=5, markeredgewidth=0)
            y += 1
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xlabel("Filtration value")


def render_landscape(dgms: list, ax, title: str = "Landscape",
                     n_landscapes: int = 3,
                     dims: tuple | None = None) -> None:
    """
    Render persistence landscapes on ax using the tent function formulation.

    dims : tuple of homology dimensions to include. None renders all.
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    active = list(range(len(dgms))) if dims is None else list(dims)
    plotted = False
    for dim in active:
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            continue
        t_min, t_max = finite[:, 0].min(), finite[:, 1].max()
        ts = np.linspace(t_min, t_max, 500)
        tents = np.maximum(
            0,
            np.minimum(
                ts[None, :] - finite[:, 0:1],
                finite[:, 1:2] - ts[None, :],
            ),
        )
        sorted_tents = np.sort(tents, axis=0)[::-1]
        color = colors[dim % len(colors)]
        for k in range(min(n_landscapes, len(sorted_tents))):
            ax.plot(ts, sorted_tents[k], color=color,
                    alpha=max(0.3, 0.9 - 0.25 * k), linewidth=1,
                    label=f'H{dim} λ{k + 1}' if k == 0 else '')
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Filtration value")
    if plotted:
        ax.legend(fontsize=6)


def render_comparison_table(rips: RipsResult, alpha: AlphaResult, ax) -> None:
    """Render a Rips vs Alpha summary metrics table into ax."""
    rips_h1_count, rips_top = h1_stats(rips.dgms)
    alpha_h1_count, alpha_top = h1_stats(alpha.dgms)
    cell_text = [
        ["simplex_count", str(rips.num_edges),     str(alpha.num_simplices)],
        ["runtime_ms",    f"{rips.elapsed_ms:.1f}", f"{alpha.elapsed_ms:.1f}"],
        ["H1_bar_count",  str(rips_h1_count),       str(alpha_h1_count)],
        ["top_persistence",
         f"{rips_top:.4f}"  if np.isfinite(rips_top)  else "—",
         f"{alpha_top:.4f}" if np.isfinite(alpha_top) else "—"],
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Rips", "Alpha"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    table.auto_set_column_width([0, 1, 2])
    ax.axis('off')
    ax.set_title("Summary  (simplex_count: Rips=edges, Alpha=all simplices)",
                 fontsize=9, pad=6)


def render_matching(d_r: np.ndarray, d_a: np.ndarray, matching: np.ndarray,
                    ax, title: str = "",
                    label_a: str = "Rips", label_b: str = "Alpha") -> None:
    """
    Overlay two H1 point sets and draw their bottleneck/Wasserstein matching lines.
    Diagonal projections indicate unmatched points mapped to the diagonal.
    """
    all_pts = (np.concatenate([d_r, d_a]) if len(d_r) and len(d_a)
               else (d_r if len(d_r) else d_a))
    lo = float(all_pts.min())
    hi = float(all_pts[np.isfinite(all_pts)].max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.5)

    if len(d_r):
        ax.scatter(d_r[:, 0], d_r[:, 1], color='tab:blue', s=30, zorder=3, label=label_a)
    if len(d_a):
        ax.scatter(d_a[:, 0], d_a[:, 1], color='tab:orange', s=30, zorder=3,
                   marker='^', label=label_b)

    for m in matching:
        i, j = int(m[0]), int(m[1])
        if i == -1:
            pt = d_a[j]
            mid = (pt[0] + pt[1]) / 2
            ax.plot([pt[0], mid], [pt[1], mid], color='gray', linestyle='--', linewidth=0.8)
        elif j == -1:
            pt = d_r[i]
            mid = (pt[0] + pt[1]) / 2
            ax.plot([pt[0], mid], [pt[1], mid], color='gray', linestyle='--', linewidth=0.8)
        else:
            p1, p2 = d_r[i], d_a[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linestyle='--', linewidth=0.8)

    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)


def render_voronoi_delaunay(pts: np.ndarray, ax,
                             circumsphere: bool = False, seed: int = 0) -> None:
    """
    Render Voronoi cells + Delaunay triangulation on ax (2D only).
    circumsphere=True overlays each triangle's circumscribed circle.
    """
    tri = Delaunay(pts)
    regions, verts, cell_colors = _build_voronoi_colors(pts, seed)
    xlim, ylim = _ax_limits(pts)

    for idx, region in enumerate(regions):
        ax.fill(*verts[region].T, color=cell_colors[idx], alpha=0.45, zorder=0)
    for idx, region in enumerate(regions):
        closed = np.vstack([verts[region], verts[region][0]])
        ax.plot(closed[:, 0], closed[:, 1], color=cell_colors[idx],
                linewidth=0.7, alpha=0.6, zorder=1)
    ax.triplot(pts[:, 0], pts[:, 1], tri.simplices, color='k', linewidth=0.6, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=12, color='k', zorder=3)

    if circumsphere:
        for simplex in tri.simplices:
            p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            cc, r2 = _circumcenter_r2(p0, p1, p2)
            if cc is None:
                continue
            ax.add_patch(Circle(cc, r2 ** 0.5, fill=False, edgecolor='crimson',
                                linewidth=0.8, alpha=0.5, linestyle='--', zorder=4))
            ax.scatter(cc[0], cc[1], color='crimson', s=10, zorder=5,
                       edgecolors='none', alpha=0.7)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    title = "Voronoi + Delaunay" + ("\n(circumspheres)" if circumsphere else "")
    ax.set_title(title)
    ax.set_aspect('equal')


def render_alpha_overlay(pts: np.ndarray, alpha_value: float, ax,
                          circumcenter: bool = False, seed: int = 0) -> None:
    """
    Render the Alpha complex on ax: Voronoi-clipped balls, alpha edges/triangles,
    optional circumcenter dots. 2D only.

    alpha_value : GUDHI units — α = r² (squared circumradius), not r.
    circumcenter=True draws a green dot at each circumcenter inside the complex
    and a red dot outside.
    """
    r = alpha_value ** 0.5
    tri = Delaunay(pts)
    regions, verts, cell_colors = _build_voronoi_colors(pts, seed)
    xlim, ylim = _ax_limits(pts)

    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()
    alpha_edges     = [s for s, f in st.get_filtration() if len(s) == 2 and f <= alpha_value]
    alpha_triangles = [s for s, f in st.get_filtration() if len(s) == 3 and f <= alpha_value]

    _n = 64
    theta = np.linspace(0, 2 * np.pi, _n, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _clip_patch(path_verts):
        n_v = len(path_verts)
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (n_v - 1) + [MplPath.CLOSEPOLY]
        vv = np.vstack([path_verts, path_verts[0]])
        patch = PathPatch(MplPath(vv, codes), visible=False)
        ax.add_patch(patch)
        return patch

    for idx, region in enumerate(regions):
        polygon = verts[region]
        cx, cy = pts[idx]
        color = cell_colors[idx]

        cell_p = _clip_patch(polygon)
        ball_p = _clip_patch(np.c_[cx + r * cos_t, cy + r * sin_t])

        disk = Circle((cx, cy), r, color=color, alpha=0.45, linewidth=0, zorder=0)
        ax.add_patch(disk)
        disk.set_clip_path(cell_p)

        arc = Circle((cx, cy), r, fill=False, edgecolor=color,
                     linewidth=1.1, alpha=0.9, zorder=2)
        ax.add_patch(arc)
        arc.set_clip_path(cell_p)

        closed = np.vstack([polygon, polygon[0]])
        (line,) = ax.plot(closed[:, 0], closed[:, 1],
                          color=color, linewidth=1.1, alpha=0.9, zorder=2)
        line.set_clip_path(ball_p)

    # Faint background structure
    ax.triplot(pts[:, 0], pts[:, 1], tri.simplices,
               color='k', linewidth=0.5, alpha=0.4, zorder=3)
    for idx, region in enumerate(regions):
        closed = np.vstack([verts[region], verts[region][0]])
        ax.plot(closed[:, 0], closed[:, 1],
                color=cell_colors[idx], linewidth=0.5, alpha=0.25, zorder=1)

    # Alpha simplices
    for i, j, k in alpha_triangles:
        t = pts[[i, j, k]]
        ax.fill(t[:, 0], t[:, 1], color='steelblue', alpha=0.25, zorder=4)
    for i, j in alpha_edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color='k', linewidth=2.0, zorder=5)

    if circumcenter:
        for simplex, filt in st.get_filtration():
            if len(simplex) != 3:
                continue
            i, j, k = simplex
            cc_pt, _ = _circumcenter_r2(pts[i], pts[j], pts[k])
            if cc_pt is None:
                continue
            if not (xlim[0] <= cc_pt[0] <= xlim[1] and ylim[0] <= cc_pt[1] <= ylim[1]):
                continue
            dot_color = 'limegreen' if filt <= alpha_value else 'tomato'
            ax.scatter(cc_pt[0], cc_pt[1], color=dot_color, s=22, zorder=6,
                       edgecolors='k', linewidths=0.4)

    ax.scatter(pts[:, 0], pts[:, 1], s=12, color='k', zorder=7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"Alpha complex  (α ≤ {alpha_value:.4f})")
    ax.set_aspect('equal')


def render_mnist_image(img: np.ndarray, ax,
                       title: str = "MNIST Digit",
                       digit: int | None = None) -> None:
    """
    Display an 8×8 grayscale digit image on ax.

    img   : (8, 8) ndarray, values 0–16
    digit : if provided, appended to the title as '(digit N)'
    """
    full_title = title if digit is None else f"{title}  (digit {digit})"
    h, w = img.shape
    ax.imshow(img, cmap='gray_r', interpolation='nearest',
              origin='upper', vmin=0, vmax=255)
    ax.set_title(full_title)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)


def render_vr_overlay(pts: np.ndarray, radius: float, ax) -> None:
    """
    Render the Vietoris-Rips complex at the given radius:
    uniform balls per point, edges for pairs within 2r, filled triangles for 3-cliques.
    """
    from scipy.spatial.distance import squareform, pdist

    n = len(pts)
    D = squareform(pdist(pts))

    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] <= 2 * radius]
    edge_set = set(edges)
    triangles = [
        (i, j, k)
        for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)
        if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set
    ]

    for pt in pts:
        ax.add_patch(Circle(pt, radius, color='tab:blue', alpha=0.15, linewidth=0, zorder=0))
        ax.add_patch(Circle(pt, radius, fill=False, edgecolor='tab:blue',
                            linewidth=0.8, alpha=0.7, zorder=1))
    for i, j, k in triangles:
        t = pts[[i, j, k]]
        ax.fill(t[:, 0], t[:, 1], color='tab:orange', alpha=0.30, zorder=2)
    for i, j in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color='k', linewidth=1.8, zorder=3)

    ax.scatter(pts[:, 0], pts[:, 1], s=12, color='k', zorder=4)
    ax.set_aspect('equal')
    ax.set_title(f"Vietoris-Rips  (r={radius:.3f})")
