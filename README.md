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
| 02    | **Complex Comparison** — Vietoris-Rips vs Alpha complexes                          | ⏳In Progress |
| 03    | **Geometric Interpretation** — Torus, MNIST digits, Betti numbers                  | 🔲 Upcoming |
| 04    | **Real Data** — RCSB PDB proteins, OpenTopography terrain                          | 🔲 Upcoming |
| 05    | **ML Integration** — Persistence images as classifier features                     | 🔲 Upcoming |
---
## Project Structure

```
tda-learning/
├── README.md
├── requirements.txt
├── portfolio/
│   └── tda-portfolio.html          # Visual project tracker
├── stage-01-pipeline-fluency/
│   └── Noisy_Circle_Comparison.ipynb
├── stage-02-complex-comparison/
├── stage-03-geometric-interpretation/
├── stage-04-real-data/
└── stage-05-ml-integration/
```
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
## Setup
**Requirements:** Python 3.8+, WSL or Linux/macOS recommended

```bash
git clone https://github.com/YOUR_USERNAME/tda-learning.git
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
