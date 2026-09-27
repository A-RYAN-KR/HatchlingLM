#!/usr/bin/env python3
"""
CLI Script to demonstrate inference-time synaptic plasticity:
1. Static Transformer weights (0 drift) vs Hebbian synaptic memory
2. Runaway naive Hebbian feedback vs Bounded homeostatic Oja's rule
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from hatchling.models.plasticity import BDHPlasticBlock, BDHOjaBlock
from hatchling.models.transformer import TransformerStaticBlock


def parse_args():
    parser = argparse.ArgumentParser(description="Test Synaptic Plasticity & Oja Homeostasis")
    parser.add_argument("--save_path", type=str, default="assets/synaptic_plasticity_drift.png", help="Save plot path")
    return parser.parse_args()


def main():
    args = parse_args()

    sample_phrase = (
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind "
        "to suffer the slings and arrows of outrageous fortune"
    )
    tokens = [ord(c) for c in sample_phrase]
    seq_len = len(tokens)
    print(f"Streaming {seq_len} tokens through plasticity models...")

    # --- Part 1: Plastic BDH vs Static Transformer ---
    embedding = nn.Embedding(256, 64)
    bdh_layer = BDHPlasticBlock(d_model=64, n_neurons=128, eta=0.08, decay=0.98)
    tf_layer = TransformerStaticBlock(d_model=64)

    initial_bdh = bdh_layer.base_synapse.clone()
    initial_tf = tf_layer.q_proj.weight.clone()
    bdh_layer.reset_synapses()

    bdh_drifts = []
    tf_drifts = []

    for tok in tokens:
        tok_vec = embedding(torch.tensor([tok])).unsqueeze(0)
        _ = bdh_layer.forward_step(tok_vec)
        _ = tf_layer.forward_step(tok_vec)

        bdh_drift = torch.norm(bdh_layer.dynamic_synapse - initial_bdh).item()
        tf_drift = torch.norm(tf_layer.q_proj.weight - initial_tf).item()
        bdh_drifts.append(bdh_drift)
        tf_drifts.append(tf_drift)

    print(f"Transformer Final Weight Drift: {tf_drifts[-1]:.6f} (Strictly 0.0 - Static)")
    print(f"BDH Synaptic State Drift:       {bdh_drifts[-1]:.4f} (Dynamically strengthened)")

    # --- Part 2: Naive Hebbian Runaway vs Stabilized Oja's Rule ---
    oja_layer = BDHOjaBlock(d_model=64, n_neurons=128, eta=0.005)
    uncon_layer = BDHPlasticBlock(d_model=64, n_neurons=128, eta=0.05, decay=0.98)

    initial_oja = oja_layer.base_synapse.clone()
    initial_uncon = uncon_layer.base_synapse.clone()
    oja_layer.reset()
    uncon_layer.reset_synapses()

    oja_drifts = []
    uncon_drifts = []

    for tok in tokens:
        tok_vec = embedding(torch.tensor([[tok]]))
        _ = oja_layer.forward_step(tok_vec)
        _ = uncon_layer.forward_step(tok_vec)

        d_oja = torch.norm(oja_layer.dynamic_synapse - initial_oja).item()
        d_uncon = torch.norm(uncon_layer.dynamic_synapse - initial_uncon).item()
        oja_drifts.append(d_oja)
        uncon_drifts.append(d_uncon)

    print(f"Max Drift under Naive Hebbian: {max(uncon_drifts):.2e} (Runaway Feedback)")
    print(f"Final Bounded Drift under Oja: {oja_drifts[-1]:.4f} (Bounded, Stable Dynamics)")

    # Render Comparison Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(range(seq_len), uncon_drifts, color="#d62728", lw=2.5, label="Naive Hebbian (Runaway)")
    axes[0].set_title("Unconstrained Hebbian Divergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Token Index", fontsize=11)
    axes[0].set_ylabel("Weight Drift (Frobenius Norm)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].plot(range(seq_len), oja_drifts, color="#2ca02c", lw=2.5, label="Oja's Rule (Homeostatic)")
    axes[1].set_title("Stabilized Synaptic Plasticity (Oja's Rule)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Token Index", fontsize=11)
    axes[1].set_ylabel("Weight Drift (Frobenius Norm)", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    plt.savefig(args.save_path, dpi=150)
    print(f"Saved plasticity drift visualization to {args.save_path}")
    plt.close()


if __name__ == "__main__":
    main()
