"""
Biological neuron sparsity analysis and visualization.
Analyzes non-negative activation distributions and percentage of silent neurons.
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
import torch


def compute_layer_sparsity(activations: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
    """
    Computes percentage of silent neurons and average active neuron count per layer.

    Args:
        activations: List of activation tensors [B, T, n_neurons] per layer.

    Returns:
        (sparsities_pct, active_counts_avg)
    """
    sparsities = []
    active_counts = []
    for act in activations:
        sparsity_pct = (act == 0.0).float().mean().item() * 100.0
        active_avg = (act > 0.0).sum(dim=-1).float().mean().item()
        sparsities.append(sparsity_pct)
        active_counts.append(active_avg)
    return sparsities, active_counts


def plot_sparsity_analysis(
    layer_sparsities: List[float],
    first_layer_act: torch.Tensor,
    tokens: List[int],
    save_path: str = "assets/bdh_activation_analysis.png"
):
    """
    Renders 2-panel figure showing layer-wise sparsity and Layer 1 activation heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Layer Sparsity Bar Plot
    axes[0].bar(
        range(1, len(layer_sparsities) + 1),
        layer_sparsities,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.85
    )
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("Layer Index", fontsize=11)
    axes[0].set_ylabel("Neuron Sparsity (% Silent)", fontsize=11)
    axes[0].set_title("Biological Sparsity Across BDH Layers", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel B: Firing Rate Heatmap
    sample_len = min(30, first_layer_act.shape[1])
    heat_data = first_layer_act[0, :sample_len, :50].detach().cpu().numpy().T
    im = axes[1].imshow(heat_data, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[1], label="Neuron Firing Rate")
    axes[1].set_title("Layer 1 Neuron Activations (Top 50 Neurons)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Character / Byte Position", fontsize=11)
    axes[1].set_ylabel("Neuron Index", fontsize=11)

    char_labels = [chr(b) if 32 <= b <= 126 else "·" for b in tokens[:sample_len]]
    axes[1].set_xticks(range(sample_len))
    axes[1].set_xticklabels(char_labels, rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
