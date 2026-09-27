#!/usr/bin/env python3
"""
CLI Script to discover monosemantic specialist neurons in HatchlingLM
and render the 3-panel interpretability dashboard.
"""

import argparse
import os
import sys
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from data.dataloader import download_dataset, load_byte_data, get_batch
from hatchling.models.bdh import HatchlingLM
from hatchling.analysis.interpretability import (
    harvest_layer_activations,
    find_top_specialist,
    plot_interpretability_dashboard
)


def parse_args():
    parser = argparse.ArgumentParser(description="Monosemantic Specialist Neuron Analysis")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hatchling_best.pt", help="Path to checkpoint")
    parser.add_argument("--save_path", type=str, default="assets/monosemantic_interpretability_dashboard.png", help="Figure save path")
    parser.add_argument("--batches", type=int, default=20, help="Number of batches to harvest")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = HatchlingLM(vocab_size=256, d_model=384, n_neurons=1536, n_layers=6, max_seq_len=128).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Checkpoint not found at '{args.checkpoint}'. Using initialized model for discovery.")

    data_path = download_dataset()
    _, val_data = load_byte_data(data_path)

    def get_batch_fn(split="val", batch_size=16, block_size=128, device="cuda"):
        return get_batch(val_data, batch_size=batch_size, block_size=block_size, device=device)

    tokens_flat, acts_flat = harvest_layer_activations(
        model=model,
        get_batch_fn=get_batch_fn,
        layer_idx=0,
        num_batches=args.batches,
        device=device
    )

    # Define syntactic and grammatical feature masks
    is_punct = np.isin(tokens_flat, [ord(c) for c in ",.:;?!'\"-"])
    is_upper = np.isin(tokens_flat, [ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    is_space = (tokens_flat == ord(" "))

    # Identify top specialist neurons
    punct_neuron, punct_ratio = find_top_specialist(acts_flat, is_punct)
    upper_neuron, upper_ratio = find_top_specialist(acts_flat, is_upper)
    space_neuron, space_ratio = find_top_specialist(acts_flat, is_space)

    print("\n" + "=" * 65)
    print(f"{'DETECTED SPECIALIST':<25} | {'NEURON ID':<12} | {'SELECTIVITY RATIO':<20}")
    print("=" * 65)
    print(f"{'Punctuation Detector':<25} | {punct_neuron:<12} | {punct_ratio:18.2f}x higher")
    print(f"{'Capitalization Detector':<25} | {upper_neuron:<12} | {upper_ratio:18.2f}x higher")
    print(f"{'Word Boundary (Space)':<25} | {space_neuron:<12} | {space_ratio:18.2f}x higher")
    print("=" * 65)

    plot_interpretability_dashboard(
        model=model,
        punct_neuron=punct_neuron,
        punct_ratio=punct_ratio,
        upper_neuron=upper_neuron,
        upper_ratio=upper_ratio,
        space_neuron=space_neuron,
        space_ratio=space_ratio,
        save_path=args.save_path,
        device=device
    )


if __name__ == "__main__":
    main()
