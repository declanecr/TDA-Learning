# tda — TDA utilities package

Shared library for the TDA-Learning project. Extracted from the stage notebooks so computation and plotting logic is written once and reused across stages.

---

## Architecture

The package is split into three layers. Each layer has one job and no upward imports.

```
tda/
├── tda_data.py      layer 1 — pure computation, no matplotlib
├── tda_viz.py       layer 2 — atomic ax-level renderers, no plt.show()
└── tda_figures.py   layer 3 — GridSpec orchestrators, owns plt.show()
```

**`tda_data`** is dependency-free of matplotlib. It generates point clouds, runs the TDA libraries, and returns typed dataclasses. Safe to import in any context (scripts, tests, headless servers).

**`tda_viz`** takes pre-computed data and a matplotlib `Axes` object and draws onto it. It never calls `plt.show()` or creates figures — it only renders.

**`tda_figures`** owns layout. Each function creates a `Figure`, wires up a `GridSpec`, calls renderers from `tda_viz`, and calls `plt.show()` at the end. These are the functions you call from a notebook cell.

---

## Importing

```python
import sys; sys.path.insert(0, '..')   # from a stage-XX/ notebook
from tda import compute_both, plot_complex_comparison
```

Everything in the public API is re-exported from `tda/__init__.py`, so a single `from tda import ...` is all that's needed.

---

## tda_data — computation layer

### Result types

| Type | Fields |
|---|---|
| `RipsResult` | `dgms`, `num_edges`, `elapsed_ms` |
| `AlphaResult` | `dgms`, `num_simplices`, `elapsed_ms` |
| `BothResult` | `pts`, `shape`, `n_points`, `noise`, `rips`, `alpha` |

### Functions

| Function | Description |
|---|---|
| `generate_point_cloud(shape, n_points, noise)` | Returns a noisy 2D/3D point cloud. `shape` is `'circle'`, `'torus'`, or `'sphere'`. |
| `compute_rips(pts, max_dim=2)` | Runs Ripser on `pts` and returns a `RipsResult`. |
| `compute_alpha(pts, max_dim=2)` | Runs GUDHI `AlphaComplex`. Filtration values are in GUDHI units: **α = r²** (squared circumradius), not `r`. Returns an `AlphaResult`. |
| `compute_both(shape, n_points, noise, max_dim=2)` | Convenience wrapper: generates a point cloud, runs both Rips and Alpha, returns a `BothResult`. |
| `filter_diagrams(dgms, threshold)` | Drops finite bars with persistence ≤ `threshold`. Infinite bars (essential classes) are always kept. Pass `threshold=0` to skip. |
| `h1_stats(dgms)` | Returns `(finite H1 bar count, max H1 persistence)`. |
| `auto_alpha_value(dgms_alpha)` | Returns the birth value of the top-persistence finite H1 bar (in GUDHI r² units). Used to pick a sensible default for geometric overlays. |
| `diagram_distances(dgms_rips, dgms_alpha)` | Computes bottleneck and Wasserstein-1 distances between Rips and Alpha diagrams per homology dimension. Returns `{dim: {'bottleneck', 'wasserstein', 'bn_match', 'ws_match', 'd_rips', 'd_alpha'}}`. |

---

## tda_viz — renderer layer

All renderers take pre-computed data and one or more `Axes`. They return `None`.

| Function | What it draws |
|---|---|
| `render_point_cloud(pts, ax, title)` | Scatter plot; 3D-aware |
| `render_persistence_diagram(dgms, ax, title)` | Persistence diagram via persim |
| `render_barcode(dgms, ax, title, dims)` | Barcode; arrow tips for infinite bars |
| `render_landscape(dgms, ax, title, n_landscapes, dims)` | Persistence landscapes (tent function) |
| `render_comparison_table(rips, alpha, ax)` | Rips vs Alpha summary metrics table |
| `render_matching(d_r, d_a, matching, ax, title, label_a, label_b)` | Bottleneck/Wasserstein matching overlay; `label_a`/`label_b` default to `"Rips"`/`"Alpha"` |
| `render_voronoi_delaunay(pts, ax, circumsphere, seed)` | Voronoi cells + Delaunay triangulation (2D) |
| `render_alpha_overlay(pts, alpha_value, ax, circumcenter, seed)` | Alpha complex with Voronoi-clipped balls (2D) |
| `render_vr_overlay(pts, radius, ax)` | Vietoris-Rips complex at given radius (2D) |

**Note on `alpha_value`:** functions that accept `alpha_value` expect GUDHI r² units, not radius. Convert with `r = alpha_value ** 0.5` when you need the geometric radius.

---

## tda_figures — figure layer

Each function creates a complete figure and calls `plt.show()`.

| Function | Description |
|---|---|
| `plot_noise_experiment(pts, rips, noise, thresholds, representations, mode)` | Multi-panel Rips-only figure. Supports `h1only`, `overlay`, and `grid` modes, single or two-threshold comparison. |
| `plot_complex_comparison(result, threshold, representations)` | 2-row (Rips / Alpha) side-by-side comparison with persistence diagrams, barcodes, and/or landscapes. |
| `plot_geometric(pts, alpha_value, shape, noise, circumsphere, circumcenter, seed)` | Voronoi+Delaunay (left) vs Alpha complex overlay (right). 2D only. |
| `plot_alpha_vr_comparison(pts, alpha_value, shape, noise, circumcenter, seed)` | Alpha complex (left) vs Vietoris-Rips (right) at the same ball radius. |
| `plot_distance_comparison(result)` | 2×2 figure: Rips PD, Alpha PD, H1 bottleneck matching, H1 Wasserstein matching. Also prints the distance table. |
| `plot_rips_comparison(result_a, result_b, label_a, label_b)` | 3×2 figure comparing Rips diagrams from two point clouds: point clouds, PDs, H1 bottleneck and Wasserstein matchings. Use this instead of `plot_distance_comparison` when comparing two different point clouds — avoids the r² vs r unit mismatch of Rips/Alpha. |
| `plot_alpha_comparison(result_a, result_b, label_a, label_b, alpha_value_a, alpha_value_b)` | 3×2 figure comparing Alpha diagrams from two point clouds. Row 0 shows the Alpha complex overlay for 2D clouds (auto-derived α) or a plain scatter for 3D. Rows 1–2 are Alpha PDs and H1 matchings. |
| `plot_full_analysis(result, threshold, alpha_value, circumcenter, seed)` | Unified 5-panel figure: point cloud, both persistence diagrams, both barcodes, summary table, and Alpha overlay. |
| `print_distance_table(distances)` | Prints a bottleneck/Wasserstein table to stdout. Called internally by `plot_distance_comparison`. |

### `representations` parameter

`plot_noise_experiment` and `plot_complex_comparison` accept a `representations` argument — an ordered list of column types to include:

| Value | Column |
|---|---|
| `'pd'` | Persistence diagram |
| `'bc'` | Barcode |
| `'pl'` | Persistence landscape |

Example: `representations=('pd', 'bc')` produces two diagram columns and omits landscapes.

### `mode` parameter (`plot_noise_experiment` only)

| Value | Behaviour |
|---|---|
| `'h1only'` | Show H₁ bars only in barcode/landscape columns |
| `'overlay'` | Show H₀ + H₁ together |
| `'grid'` | 2×2 sub-grid: H₁ vs H₀+H₁ × unfiltered vs filtered |

---

## Quick-start examples

```python
import sys; sys.path.insert(0, '..')
from tda import compute_both, plot_complex_comparison, plot_full_analysis

# Generate data and compute both complexes
result = compute_both('circle', n_points=200, noise=0.05)

# Rips vs Alpha side-by-side
plot_complex_comparison(result, threshold=0.05)

# Unified analysis panel
plot_full_analysis(result, threshold=0.05)
```

```python
from tda import compute_rips, generate_point_cloud, plot_noise_experiment

pts  = generate_point_cloud('circle', n_points=150, noise=0.08)
rips = compute_rips(pts)

plot_noise_experiment(
    pts, rips, noise=0.08,
    thresholds=[0.0, 0.03],
    representations=('pd', 'bc'),
    mode='h1only',
)
```
```
from tda import generate_point_cloud, render_point_cloud

torus = generate_point_cloud('torus', n_points=300, noise=0.05)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
render_point_cloud(torus, ax, title="torus")
plt.show()
```
<img width="410" height="421" alt="image" src="https://github.com/user-attachments/assets/4e8f2ce3-e764-4e96-bada-23a91b7d4698" />




---

## Dependencies

| Library | Used in |
|---|---|
| `numpy` | all layers |
| `ripser` | `tda_data` — Vietoris-Rips persistence |
| `gudhi` | `tda_data`, `tda_viz` — Alpha complex |
| `persim` | `tda_data`, `tda_viz` — diagram distances, plotting |
| `tadasets` | `tda_data` — point cloud generation |
| `scipy` | `tda_viz` — Delaunay, Voronoi |
| `matplotlib` | `tda_viz`, `tda_figures` |
