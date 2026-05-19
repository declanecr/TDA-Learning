"""
tda — Topological Data Analysis utilities.

Three-layer architecture
------------------------
tda.tda_data    : pure computation — generate, compute, filter, stats
tda.tda_viz     : atomic ax-based renderers — no computation, no plt.show()
tda.tda_figures : GridSpec figure orchestrators — own layout and plt.show()

Quick start
-----------
    import sys; sys.path.insert(0, '..')
    from tda import compute_both, plot_complex_comparison

    result = compute_both('circle', n_points=200, noise=0.05)
    plot_complex_comparison(result, threshold=0.05)
"""

from .tda_data import (
    RipsResult,
    AlphaResult,
    BothResult,
    CubicalResult,
    MNISTResult,
    generate_point_cloud,
    compute_rips,
    compute_alpha,
    compute_both,
    compute_cubical,
    compute_mnist,
    filter_diagrams,
    h1_stats,
    auto_alpha_value,
    diagram_distances,
)

from .tda_viz import (
    render_point_cloud,
    render_persistence_diagram,
    render_barcode,
    render_landscape,
    render_comparison_table,
    render_matching,
    render_voronoi_delaunay,
    render_alpha_overlay,
    render_vr_overlay,
    render_mnist_image,
)

from .tda_figures import (
    plot_noise_experiment,
    plot_complex_comparison,
    plot_geometric,
    plot_alpha_vr_comparison,
    plot_distance_comparison,
    plot_rips_comparison,
    plot_alpha_comparison,
    plot_full_analysis,
    print_distance_table,
    plot_mnist_analysis,
)

__all__ = [
    # data
    "RipsResult", "AlphaResult", "BothResult", "CubicalResult", "MNISTResult",
    "generate_point_cloud", "compute_rips", "compute_alpha", "compute_both",
    "compute_cubical", "compute_mnist",
    "filter_diagrams", "h1_stats", "auto_alpha_value", "diagram_distances",
    # viz
    "render_point_cloud", "render_persistence_diagram",
    "render_barcode", "render_landscape",
    "render_comparison_table", "render_matching",
    "render_voronoi_delaunay", "render_alpha_overlay", "render_vr_overlay",
    "render_mnist_image",
    # figures
    "plot_noise_experiment", "plot_complex_comparison",
    "plot_geometric", "plot_alpha_vr_comparison",
    "plot_distance_comparison", "plot_rips_comparison", "plot_alpha_comparison",
    "plot_full_analysis", "print_distance_table",
    "plot_mnist_analysis",
]
