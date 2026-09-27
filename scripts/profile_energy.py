#!/usr/bin/env python3
"""
CLI script for Neuromorphic Energy and Sparsity Profiling (SynOps vs FLOPs).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hatchling.analysis.energy import compute_energy_metrics, print_energy_table, plot_energy_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Neuromorphic Energy Profiling")
    parser.add_argument("--density", type=float, default=0.1406, help="Empirical active neuron density (1 - sparsity)")
    parser.add_argument("--save_path", type=str, default="assets/energy_profiling.png", help="Figure save path")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = compute_energy_metrics(empirical_density=args.density)
    print_energy_table(metrics)
    plot_energy_comparison(metrics, save_path=args.save_path)
    print(f"Energy comparison chart saved to: {args.save_path}")


if __name__ == "__main__":
    main()
