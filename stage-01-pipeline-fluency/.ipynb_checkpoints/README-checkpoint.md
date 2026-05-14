# Project 1 — Noisy Circle TDA

Explores **Topological Data Analysis (TDA)** on a synthetic noisy circle point cloud. The notebook uses **Vietoris–Rips persistent homology** (via `ripser`) to detect the underlying circular topology and demonstrates how persistence filtration thresholds can separate signal from noise.

## What the notebook does

### 1. Baseline TDA (cells 1–2)
Generates a 100-point noisy circle sampled uniformly from [0, 2π] with Gaussian noise (σ = 0.1), computes its persistent homology, and displays:
- The raw point cloud scatter plot
- A persistence diagram (birth vs. death for H₀ and H₁ features)
- A manual H₁ barcode plot

### 2. Barcode utility (`plot_barcodes`, cell 3)
A reusable function that renders barcodes either **side-by-side** (one subplot per homology dimension) or **overlaid** in a single plot with colour-coded dimensions.

### 3. Experiment framework (`compute_circle` + `plot_experiment_pair`, cells 4–5)
The core of the notebook. Two functions that together run controlled experiments:

- **`compute_circle(n_points, noise, max_dim)`** — generates a noisy circle point cloud and returns it alongside its persistence diagrams.
- **`plot_experiment_pair(...)`** — produces a composite figure with the point cloud on the left and any combination of three TDA visualisations on the right:
  - **Persistence diagram (PD)** — birth/death scatter with diagonal
  - **Barcode (BC)** — horizontal bars per feature
  - **Persistence landscape (PL)** — top-3 landscape functions (via `persim`)

  Pass `thresholds` as a single value to get one row, or as `[unfiltered, filtered]` to get a two-row comparison. Use `overlay=True` to include H₀ alongside H₁, and `grid=True` to split each column into an H₁-only and H₀+H₁ subplot. Figures are auto-saved to `tda_outputs/` with descriptive filenames.

### 4. Sweep experiments (cell 6)
Runs the experiment across increasing noise levels (0.0 → 0.5) for n = 50 and n = 100, generating a grid of output figures that show how noise degrades the dominant H₁ feature (the loop) and how filtration thresholds recover it.

## Dependencies

```
numpy
ripser
persim
matplotlib
```

Install with:
```bash
pip install numpy ripser persim matplotlib
```

## Output

All figures are saved to `tda_outputs/` with filenames encoding the configuration:
```
circle_n<points>_ns<noise>_<diagrams>_th<threshold>[_ovl][_grid].png
```

Pre-generated comparison figures (varying sample size and noise level) are also included at the project root.
