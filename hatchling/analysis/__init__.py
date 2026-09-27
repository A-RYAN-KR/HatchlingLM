"""
Empirical analysis and benchmark suite for HatchlingLM.
Includes sparsity, neuromorphic energy profiling, graph reasoning, and monosemantic interpretability.
"""

from hatchling.analysis.sparsity import compute_layer_sparsity, plot_sparsity_analysis
from hatchling.analysis.energy import compute_energy_metrics, print_energy_table, plot_energy_comparison
from hatchling.analysis.graph_benchmark import generate_graph_sample, collate_graph_batch, evaluate_graph_accuracy
from hatchling.analysis.interpretability import harvest_layer_activations, find_top_specialist, plot_interpretability_dashboard

__all__ = [
    "compute_layer_sparsity",
    "plot_sparsity_analysis",
    "compute_energy_metrics",
    "print_energy_table",
    "plot_energy_comparison",
    "generate_graph_sample",
    "collate_graph_batch",
    "evaluate_graph_accuracy",
    "harvest_layer_activations",
    "find_top_specialist",
    "plot_interpretability_dashboard",
]
