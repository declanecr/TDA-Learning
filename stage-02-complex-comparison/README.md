# P2-Complexes: Rips vs Alpha Complex Comparison

A notebook for building intuition about the two most common simplicial complex constructions in TDA — Vietoris-Rips and Alpha — by computing, visualising, and directly comparing them on the same point cloud.

---

## What it does

Given a noisy point cloud sampled from a known shape (circle, torus, sphere), the notebook:

1. Builds a **Vietoris-Rips complex** (via Ripser) and an **Alpha complex** (via GUDHI) over the same points
2. Computes **persistence diagrams** for both, up to H2
3. Visualises the Alpha complex geometrically — showing how it is constructed from the intersection of balls with Voronoi cells (the Nerve theorem in action)
4. Compares the two filtrations quantitatively: simplex count, runtime, H1 bar count, top persistence, and bottleneck/Wasserstein distances between diagrams

The central question it answers: **do Rips and Alpha recover the same topology, and at what cost?**

---

## Background concepts

| Concept | What it means here |
|---|---|
| **Filtration** | A nested sequence of simplicial complexes built by slowly growing a scale parameter α |
| **Vietoris-Rips** | Add a simplex whenever all pairwise distances between its vertices are ≤ 2α. Computationally simple but produces large complexes |
| **Alpha complex** | Add a simplex only when the corresponding Voronoi cells intersect within balls of radius r = √α. Provably homotopy-equivalent to the union of balls (Nerve theorem), and much smaller |
| **Persistence diagram** | Records (birth, death) filtration values for each topological feature. Points far from the diagonal = long-lived = likely real signal |
| **H0 / H1 / H2** | Connected components / loops / voids. A circle has one persistent H1 bar; a torus has two |
| **Bottleneck / Wasserstein distance** | Metrics between diagrams. Small distance = both filtrations detected the same topology |

---

## How it works — code structure

The notebook is split into two code cells (plus usage cells).

### Cell 1 — data layer + general renderers

```
compute_comparison(shape, n_points, noise)
    → (pts, dgms_rips, num_edges, rips_ms,
       dgms_alpha, num_simplices, alpha_ms, noise)
```
Single source of truth for all data. Generates the point cloud with `tadasets`, runs Ripser for Rips persistence, runs GUDHI AlphaComplex for Alpha persistence.

**Helper utilities**

| Function | Purpose |
|---|---|
| `_filter_dgms(dgms, threshold)` | Strip finite bars below a persistence threshold (noise suppression) |
| `_h1_stats(dgms)` | Extract finite H1 bar count and max persistence from a diagram list |

**Ax-based renderers** — each accepts a `matplotlib.Axes` and draws exactly one panel into it:

| Function | Draws |
|---|---|
| `render_point_cloud(pts, ax)` | Scatter plot; uses `projection='3d'` automatically for 3D pts |
| `render_persistence_diagrams(dgms_rips, dgms_alpha, ax_r, ax_a, ...)` | Rips and Alpha persistence diagrams side by side |
| `render_barcode_comparison(dgms_rips, dgms_alpha, ax_r, ax_a)` | Rips and Alpha barcodes side by side |
| `render_comparison_table(dgms_rips, ..., dgms_alpha, ..., ax)` | 4-row stats table rendered via `ax.table()` |
| `plot_barcode(dgms, ax)` | Barcode for a single diagram list |
| `plot_landscape(dgms, ax)` | Persistence landscape (sorted tent functions) |

**Figure-creating functions** — own their `plt.figure()` calls:

| Function | Produces |
|---|---|
| `plot_comparison(pts, ...)` | 2-row (Rips / Alpha) × N-column figure; columns selectable via `show_pd`, `show_bc`, `show_pl` |
| `compare_diagrams(dgms_rips, dgms_alpha, ...)` | Side-by-side diagram figure + bottleneck/Wasserstein table printed to stdout; returns distance dict |

### Cell 2 — geometric layer + main entry point

**Geometric utilities**

| Function | Purpose |
|---|---|
| `_circumcenter_r2(p0, p1, p2)` | Circumcenter and squared circumradius of a triangle |
| `_voronoi_finite_polygons_2d(vor)` | Extends infinite Voronoi ridges to finite polygons for plotting |
| `_build_voronoi_colors(pts, seed)` | Shared Voronoi tessellation + HSV cell colours used by both geometric renderers |
| `_ax_limits(pts)` | Padded axis bounds |
| `_auto_alpha_value(dgms_alpha)` | Picks a data-driven default α for the overlay panel (see below) |

**Geometric renderers** (ax-based, 2D only):

| Function | Draws |
|---|---|
| `_render_voronoi_delaunay(pts, ax)` | Full Voronoi cells (colour-coded) + Delaunay triangulation |
| `render_alpha_overlay(pts, alpha_value, ax)` | Ball ∩ Voronoi intersection per point; filled alpha triangles; bold alpha edges; optional circumcenters |

**Figure-creating functions:**

| Function | Produces |
|---|---|
| `plot_delaunay_alpha(pts, alpha_value, ...)` | 2-panel figure: Voronoi+Delaunay \| Alpha overlay — for exploring a single α value |
| `run_full_analysis(shape, n_points, noise, ...)` | Unified 5-panel figure (see below) |

---

## Entry points

### `run_full_analysis` — the main view

```python
run_full_analysis('circle', n_points=200, noise=0.05)
run_full_analysis('torus',  n_points=300, noise=0.05, threshold=0.02, save=True)
```

Produces a single 16×10 figure with five panels:

```
┌─────────────┬──────────────┬──────────────┐
│             │   Rips PD    │   Alpha PD   │
│ Point Cloud ├──────────────┼──────────────┤
│             │  Rips BC     │  Alpha BC    │
├─────────────┼──────────────┴──────────────┤
│   Alpha     │                             │
│   overlay   │      Summary table          │
└─────────────┴─────────────────────────────┘
```

Parameters:

| Parameter | Default | Effect |
|---|---|---|
| `shape` | — | `'circle'` (2D), `'torus'` (3D), `'sphere'` (3D) |
| `n_points` | — | Number of sample points |
| `noise` | — | Gaussian noise standard deviation |
| `threshold` | `0.0` | Minimum persistence to display in diagrams and barcodes |
| `alpha_value` | auto | GUDHI α = r² for the overlay panel; auto-derived if `None` |
| `circumcenter` | `False` | Show triangle circumcenters on overlay (green = in complex, red = not) |
| `seed` | `0` | RNG seed for Voronoi cell colours |
| `save` | `False` | Write PNG to `tda_outputs/` |

Returns a dict `{'pts', 'dgms_rips', 'dgms_alpha', 'alpha_value_used'}` for further analysis.

**How `alpha_value` is chosen automatically:** `_auto_alpha_value` finds the H1 bar in the Alpha diagram with the largest persistence (death − birth) and returns its *birth* filtration value. This is the smallest α at which the dominant topological loop first appears — so the overlay shows the complex at the moment it first detects the main signal.

---

### `run_comparison` — Rips vs Alpha side by side

```python
run_comparison('circle', n_points=200, noise=0.10, show_pd=True, show_bc=True)
```

Renders the 2-row Rips/Alpha comparison figure and prints the summary table to stdout. Accepts the same `threshold`, `save` parameters. Columns displayed are controlled by `show_pd` / `show_bc` / `show_pl`; if none are set all three are shown.

---

### `run_geometric` — step through filtration values

```python
run_geometric('circle', n_points=50, noise=0.1,
              alpha_values=[0.01, 0.05, 0.1, 0.3, 0.7],
              circumcenter=True)
```

Renders one 2-panel Voronoi+Alpha figure per value in `alpha_values`, using the same point cloud each time. Good for watching the complex grow and building intuition about what α controls.

`alpha_value` is in GUDHI's internal units: **α = r²** (squared circumradius), not r. A circle with r ≈ 0.3 corresponds to α = 0.09.

---

### `run_distance_comparison` — quantify diagram similarity

```python
run_distance_comparison('circle', n_points=200, noise=0.05)
```

Computes bottleneck and Wasserstein-1 distances between the Rips and Alpha diagrams for each homology dimension, prints a table, and shows both diagrams side by side. Returns `{dim: {'bottleneck': float, 'wasserstein': float}}`.

---

### `plot_rips_comparison` — compare two point clouds via Rips

```python
from tda import compute_both, plot_rips_comparison

a = compute_both('circle', n_points=200, noise=0.1)
b = compute_both('circle', n_points=200, noise=0.1)
plot_rips_comparison(a, b, label_a="Circle A", label_b="Circle B")
```

Compares the Rips persistence diagrams of two different point clouds directly. Produces a 3×2 figure:

```
┌──────────────┬──────────────┐
│  Point Cloud │  Point Cloud │
│      A       │      B       │
├──────────────┼──────────────┤
│  Rips PD A   │  Rips PD B   │
├──────────────┼──────────────┤
│ H1 Bottleneck│ H1 Wasserstein│
│   matching   │   matching   │
└──────────────┴──────────────┘
```

Also prints a bottleneck/Wasserstein distance table to stdout.

**Why Rips vs Rips instead of `plot_distance_comparison`?** The Alpha complex filtration uses GUDHI's internal units where α = r² (squared circumradius), while Rips uses the Euclidean radius r. Comparing their diagrams directly mixes scales, making the resulting distances geometrically uninterpretable. Use `plot_rips_comparison` whenever you want to compare two point clouds — it keeps both diagrams in the same filtration units.

---

### `plot_alpha_comparison` — compare two point clouds via Alpha

```python
plot_alpha_comparison(a, b, label_a="Circle A", label_b="Circle B")
```

The Alpha-side counterpart to `plot_rips_comparison`. Compares the Alpha persistence diagrams of two point clouds, which are geometrically valid to compare directly (both in r² units). Produces the same 3×2 layout, but with a key difference in row 0: for 2D point clouds the top row shows the actual **Alpha complex geometric overlay** instead of a plain scatter, giving visual intuition about why the diagrams look the way they do.

```
┌──────────────────┬──────────────────┐
│  Alpha overlay A │  Alpha overlay B │  ← 2D only; plain scatter for 3D
│  (α auto-derived)│  (α auto-derived)│
├──────────────────┼──────────────────┤
│  Alpha PD A      │  Alpha PD B      │
├──────────────────┼──────────────────┤
│  H1 Bottleneck   │  H1 Wasserstein  │
│  matching        │  matching        │
└──────────────────┴──────────────────┘
```

`alpha_value_a` / `alpha_value_b` default to `None` — the α is auto-derived from the top-persistence H1 bar in each diagram (GUDHI r² units).

---

## What you can learn from using it

**Rips and Alpha recover the same topology.**
For a clean circle (low noise), both methods find exactly one persistent H1 bar. The bottleneck distance between their diagrams is small — typically under 0.05. This is the theoretical guarantee: Alpha is a subcomplex of the Rips complex at matching scale, and both are homotopy-equivalent to the union of balls.

**Alpha is dramatically more efficient.**
Rips builds a complex on all pairwise distances; the simplex count grows combinatorially. Alpha uses the Delaunay triangulation to limit which simplices can appear, producing far fewer simplices. For n=200 points on a circle you might see ~4000 Rips edges vs ~600 Alpha simplices (all dimensions combined) with nearly identical diagrams.

**The geometric view explains *why* Alpha works.**
`run_geometric` shows the Nerve theorem concretely: each point's contribution to the complex is the intersection of its ball with its Voronoi cell. Two points share an edge when their balls overlap inside their shared Voronoi ridge. Three points form a triangle only when all three balls meet at a common point. This is exactly the circumcenter condition — visible when `circumcenter=True`.

**Noise tolerance is similar, but manifests differently.**
With high noise, both methods accumulate many short-lived features near the diagonal. Applying `threshold` removes these. The `top_persistence` row in the summary table tracks the dominant signal; as long as it stays large relative to noise, the topology is recoverable.

**H1 bar count is a useful sanity check.**
A circle should have one persistent H1 bar. A torus should have two (one for each independent loop). If both methods agree on the count and the bars are far from the diagonal, you can be confident the topology is real. If counts disagree, the noise level or point density is likely insufficient.

---

## Dependencies

```
ripser        # Vietoris-Rips persistence
gudhi         # Alpha complex + simplex tree
persim        # Diagram plotting, bottleneck, Wasserstein
tadasets      # Point cloud generators (circle, torus, sphere)
scipy         # Delaunay triangulation, Voronoi diagram
numpy
matplotlib
```
