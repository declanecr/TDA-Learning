# TDA-Learning
A self-directed curriculum for research in **Topological Data Analysis (TDA)** and **Persistent Homology**-built project-first with comprehension checkpoints after each stage
> Shape is a language. This is how I'm learning to read it.
---
## What this is
TDA extracts shape-based features from data using tools from algebraic topology. Instead of asking _"how close are points?"_ it asks _"what loops, voids, and connected components as we grow a neighborhood radius?"_  <br>

This repo documents a five-stage learning roadmap — from pipeline basics through real-data experiments and ML integration.

---
## Roadmap

| Stage | Topic                                                                              | Status      |
| ----- | ---------------------------------------------------------------------------------- | ----------- |
| 01    | **Pipeline Fluency** — Noisy circle, Ripser, H₀/H₁, barcodes, persistence diagrams | ✅ Complete  |
| 02    | **Complex Comparison** — Vietoris-Rips vs Alpha complexes                          | ✅ Complete  |
| 03    | **Geometric Interpretation** — Torus, MNIST digits, Betti numbers                  | ⌛ In Progress
| 04    | **Real Data** — RCSB PDB proteins, OpenTopography terrain                          | 🔲 Upcoming |
| 05    | **ML Integration** — Persistence images as classifier features                     | 🔲 Upcoming |
---
## Project Structure

```
tda-learning/
├── README.md
├── requirements.txt
├── tda/                            # Shared TDA utilities package
│   ├── __init__.py                 # Re-exports full public API
│   ├── tda_data.py                 # Pure computation layer (no matplotlib)
│   ├── tda_viz.py                  # Atomic ax-level renderers
│   └── tda_figures.py              # GridSpec figure orchestrators
├── portfolio/
│   └── tda-portfolio.html          # Visual project tracker
├── stage-01-pipeline-fluency/
│   └── Noisy_Circle_Comparison.ipynb
├── stage-02-complex-comparison/
├── stage-03-geometric-interpretation/
├── stage-04-real-data/
└── stage-05-ml-integration/
```

The `tda/` package is a three-layer library extracted from the notebooks. All stage notebooks import from it rather than re-implementing computation or plotting inline. See [`tda/README.md`](tda/README.md) for a full API reference.
---
## Stage 01 — Noist Circle $H_1$ Recovery
**Core question:** Can persistent homology recover the circular structure of a point cloud under increasing noise?

**What it does:**
- Generates noisy samples from $S^1$ (a circle in 2D)
- Runs Vietoris-Rips filtration via Ripser
- Computes $H_0$ (connected components) and $H_1$ (loops)
- Visualizes persistence diagrams and barcodes
- Sweeps noise levels to find the persistnece threshold at which the $H_1$ signal degrades

**Key insights:** 
- One long bar in $H_1$ = one persistent loop = the circle. Everything else is noise.
    <img width="1971" height="601" alt="circle_n100_ns0 1_pd+bc_th0 0" src="https://github.com/user-attachments/assets/cf51658f-be46-4c0e-8b97-88c4031e4b75" />

- **Hierarchy of Variables:**
  - Sample Size determines the foundation
    - As sample size increases, the gaps between points become smaller (higher density):
      - denser grid picks up signal _sooner_ &rarr; shifts the birth value left
      - triangles which _fill in the loop_ appear sooner &rarr; shifts death value left
        - death drops _more slowly_ than birth &rarr; **persistence increases with sample size**
    
    <img width="705" height="1156" alt="Birth at different sample size" src="https://github.com/user-attachments/assets/d6bb892e-fa83-45b2-926b-e67edb7ba5fc" />
  - Noise determines the signal quality
    - As noise level increases, more points scatter inward/outward of the circle &rarr; pairwise gaps become irregular:
      - some "shortcut edge" appears at a **smaller** $\epsilon$ than it would on a clean circle &rarr; kills cycle earlier &rarr; lower death value 
      - scattered points means loops form later &rarr; birth rises slightly  
      - **more noise &rarr; lower persistence**
    <img width="705" height="1443" alt="Death at different noise values" src="https://github.com/user-attachments/assets/56aea7fa-a92e-4c4a-8569-2c45c9b96862" />
  - Threshold permanently loses data and is used for visual interpretation ONLY
- **Birth/Death** are decoupled — birth does NOT determine death (they happen at _independent_ filtration values)
  - **BIRTH** reflects **SAMPLING**  
  - **DEATH** reflects **GEOMETRY**
  - if circle radius doubles &rarr; birth AND death values double
  - low sample size can mean true loop never closes cleanly
    - small sample size INCREASES the birth value (doesn't affect death) — sparse point cloud's require larger radius for filtration construction
  - noise creates many short-spanned spurious loops
    - _High Noise + Low `n`_ &rarr; noisy features can OUTIVE signal feature. ONLY FIX IS MORE DATA
      <img width="1962" height="601" alt="circle_n50_ns0 05_pd+bc_th0 0" src="https://github.com/user-attachments/assets/716289ad-a62a-47df-92f2-c87d747bf037" />
      <img width="1962" height="601" alt="circle_n50_ns0 5_pd+bc_th0 0" src="https://github.com/user-attachments/assets/c959a774-0756-4e66-af85-47d352133a4c" />

---
## Stage 02 — Vietoris-Rips vs Alpha Complex Comparison

**Core question:** Do complex choice and simplex count matter if the persistence diagrams are (nearly) the same?

**What it does:**
- Generates shared point clouds (noisy circle, torus) used by both complexes
- Computes Rips PH via Ripser; Alpha PH via Gudhi
- Compares simplex counts, runtimes, and filtration values
- Inspects the Delaunay triangulation underlying Alpha using `scipy.spatial`
- Quantifies diagram similarity via bottleneck and Wasserstein distances
---
### How they're built

**Vietoris-Rips** adds an edge between two points whenever their distance ≤ ε, and fills in higher simplices whenever all pairwise edges exist. It has no awareness of geometry — it checks *all* pairwise distances. This causes simplex counts to grow as $O(n^k)$ for $k$-dimensional complexes.

**Alpha** is constrained to the Delaunay triangulation. A simplex is only added at filtration value $\alpha$ if its circumsphere radius² ≤ $\alpha$ *and* the circumsphere contains no other points (the **empty circumsphere condition**). This geometric grounding keeps simplex counts at $O(n)$ in 2D/3D.

> **Analogy:** Rips inflates a balloon uniformly around every point with no awareness of neighbours. Alpha is shrink-wrap — it only grows where the geometry says points are genuinely adjacent.

<img width="865" height="1230" alt="Screenshot From 2026-05-18 13-43-11" src="https://github.com/user-attachments/assets/d5d28afa-f858-4c62-b122-b66539b767e2" />
<img width="865" height="1230" alt="Screenshot From 2026-05-18 13-43-40" src="https://github.com/user-attachments/assets/c8b6b7ba-abe8-4913-9002-9977568328b3" />

---

### The Delaunay triangulation is Alpha's backbone

Alpha cannot add any simplex that doesn't already exist in the Delaunay triangulation. At each filtration step, a Delaunay simplex is *admitted* only if its circumsphere has been reached. This means:
- Alpha is a strict **subset** of the Delaunay triangulation at every ε
- Rips can produce "geometrically impossible" simplices — triangles whose circumcircles contain other points — because it applies no such constraint

<img width="865" height="1230" alt="image" src="https://github.com/user-attachments/assets/9fc4ad5a-d883-442a-8391-cf2537a0e603" />
<img width="865" height="1230" alt="image" src="https://github.com/user-attachments/assets/1c8e7be7-8090-4acb-b29c-344860c3b707" />




---

### Simplex count comparison

Rips simplex counts grew dramatically with sample size; Alpha grew near-linearly:

| Dataset        | n   | Rips edges | Alpha simplices | Ratio  |
| -------------- | --- | ---------- | --------------- | ------ |
| Noisy circle   | 100 | 4462         | 555               | ~8×    |
| Noisy circle   | 200 | 17154         | 1149               | ~15× |
| Noisy circle   | 300 | 39710          | 1747               | ~23×   |
| Torus          | 300 | 94044          | 12783               | ~7.4×  |


<!-- IMAGE: Bar chart or line plot of simplex count vs n for both complexes -->

The downstream consequence: more simplices → larger boundary matrices → slower matrix reduction (the core PH algorithm) → higher RAM usage. Rips becomes computationally impractical before Alpha does on the same dataset.

---

### Filtration values are not directly comparable

Rips filtration values are raw pairwise distances. Gudhi's Alpha filtration values are **squared circumradii** ($\alpha = r^2$), which is why they appear much smaller. Don't compare them numerically without accounting for this scaling.

---

### Despite different structures, diagrams are nearly identical

The **Nerve Theorem** guarantees this: both complexes are valid approximations of the same underlying topological space (the union of balls at radius ε). Any "good cover" of that space yields the same persistent homology — so despite different simplex sets, they're triangulating the same shape.

This was verified quantitatively using:
- **Bottleneck distance** — smallest possible worst-case matched pair displacement between diagrams
- **Wasserstein distance** — total displacement across all matched pairs (not just the worst one)

A bottleneck distance much smaller than the signal bar's persistence confirms the diagrams are functionally equivalent.

_NOTE:_ Alpha complex PD's have a different scale than Rips because $\alpha = r^2$, therefore the bottleneck and wasserstein distances in this comparison will indicate that the diagrams are very different, despite them conveying almost identical information about the same point cloud.

#### Circle
<img width="1209" height="1011" alt="Circle rips vs alpha W B distances" src="https://github.com/user-attachments/assets/b2af1457-3af2-4106-a73c-ec3159f4e018" />

#### Torus
<img width="1197" height="1011" alt="Torus rips vs alpha W B distances" src="https://github.com/user-attachments/assets/ff472513-024c-4006-b6a2-7105f0c6e5b7" />

---

### Torus topology recovery

The torus has known Betti numbers: $\beta_0 = 1$, $\beta_1 = 2$, $\beta_2 = 1$.

| Feature | Meaning                                      |
| ------- | -------------------------------------------- |
| $\beta_0 = 1$ | One connected component                |
| $\beta_1 = 2$ | Two independent loops (short way and long way around the donut) |
| $\beta_2 = 1$ | One enclosed void (the interior of the tube surface) |

Both complexes recovered this signature. Note: the torus Rips/Alpha ratio (~7.4×) is lower than the circle at the same sample size (23x)— the torus is a 2D surface in 3D, so the Delaunay triangulation is still sparse relative to what Rips constructs, but less aggressively so than on a 1D curve.

<img width="1611" height="911" alt="Torus topology recovery - NO threshold" src="https://github.com/user-attachments/assets/6dd095f9-48f7-4286-9fa6-a4201c20249e" />
<img width="1611" height="911" alt="Torus topology recovery - Threshold" src="https://github.com/user-attachments/assets/0c115631-1ea9-4854-8db6-3225dcf22196" />

---

### When to use which complex

| Criterion              | Use Alpha                         | Use Rips                         |
| ---------------------- | --------------------------------- | -------------------------------- |
| Dimensionality         | 2D or 3D data                     | Any dimension                    |
| Data type              | Geometric / spatial               | Abstract / high-dimensional      |
| Sample size            | Large $n$ (efficiency matters)    | Small–medium $n$                 |
| Filtration values      | Need geometric interpretability   | Distance-based is sufficient     |
| Implementation         | Gudhi                             | Ripser                           |

> **Limit of Alpha:** Delaunay triangulation in high dimensions is computationally intractable. For data beyond ~3D, Rips is the default.

---

### Key takeaways

1. **Same topology, different structure** — Rips and Alpha are two roads to the same destination. The Nerve Theorem ensures they agree on what matters.
2. **The simplex count gap is large and grows** — Alpha's geometric constraint keeps it $O(n)$; Rips blows up as $O(n^k)$.
3. **Filtration values are not interchangeable** — Rips uses distances, Alpha uses squared circumradii.
4. **Alpha is Delaunay-constrained** — it cannot add geometrically unjustified simplices. Rips can and does.
5. **The persistence diagram is a record, not a live object** — features are born and die *during* filtration; the diagram captures when, not what's currently active.
6. **High-dimensional data breaks Alpha** — Rips is the universal fallback when geometry can't be leveraged.

---
## Setup
**Requirements:** Python 3.8+, WSL or Linux/macOS recommended

```bash
git clone https://github.com/declanecr/tda-learning.git
cd tda-learning
pip install -r requirements.txt
jupyter lab
```

**Core libraries:**

```
ripser
persim
gudhi
scikit-learn
scipy
numpy
matplotlib
```
---
## Key concepts in scope

- **Simplicial complexes** — Vietoris-Rips, Alpha, Čech
- **Filtration** — growing the neighborhood radius ε from 0 → ∞
- **Persistent homology** — tracking when topological features are born and die
- **Betti numbers** — β₀ (components), β₁ (loops), β₂ (voids)
- **Persistence diagrams & barcodes** — the standard TDA output
- **Cycle representatives** — which actual data points form a loop
- **Persistence images** — vectorised PH output for ML pipelines
- **Nerve Theorem** — why Vietoris-Rips captures topology at all
---
## Resources
- [Ripser.py docs](https://ripser.scikit-tda.org/)
- [Gudhi library](https://gudhi.inria.fr/)

---
_Built with curiosity and persistent homology._
