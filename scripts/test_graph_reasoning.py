#!/usr/bin/env python3
"""
CLI Script to train and benchmark HatchlingLM vs Transformer on
algorithmic multi-step directed graph reachability reasoning.
"""

import argparse
import os
import sys
import random

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from hatchling.models.bdh import HatchlingLM
from hatchling.models.transformer import StandardTransformer
from hatchling.analysis.graph_benchmark import (
    generate_graph_sample,
    collate_graph_batch,
    evaluate_graph_accuracy
)


def parse_args():
    parser = argparse.ArgumentParser(description="Graph Reachability Reasoning Benchmark")
    parser.add_argument("--steps", type=int, default=250, help="Fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--train_samples", type=int, default=600, help="Number of synthetic train samples")
    parser.add_argument("--val_samples", type=int, default=150, help="Number of synthetic val samples")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Generate synthetic dataset
    train_samples = [generate_graph_sample(chain_len=2) for _ in range(args.train_samples)]
    val_samples = [generate_graph_sample(chain_len=2) for _ in range(args.val_samples)]
    print(f"Sample Graph Query:\n{train_samples[0][0]}\n")

    # Instantiate models
    model_bdh = HatchlingLM(vocab_size=256, d_model=384, n_neurons=1536, n_layers=6, max_seq_len=128).to(device)
    model_tf = StandardTransformer(vocab_size=256, d_model=384, n_heads=6, d_ff=1536, n_layers=6, max_seq_len=128).to(device)

    opt_bdh = torch.optim.AdamW(model_bdh.parameters(), lr=args.lr)
    opt_tf = torch.optim.AdamW(model_tf.parameters(), lr=args.lr)

    print(f"Training BDH vs Transformer on Graph Path Reasoning ({args.steps} steps)...")
    for _ in range(args.steps):
        batch_indices = random.sample(range(len(train_samples)), args.batch_size)
        sub = [train_samples[i] for i in batch_indices]
        bx, by = collate_graph_batch(sub, device=device)

        # BDH step
        opt_bdh.zero_grad()
        if is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss_bdh = model_bdh(bx, targets=by)
        else:
            _, loss_bdh = model_bdh(bx, targets=by)
        loss_bdh.backward()
        opt_bdh.step()

        # Transformer step
        opt_tf.zero_grad()
        if is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss_tf = model_tf(bx, targets=by)
        else:
            _, loss_tf = model_tf(bx, targets=by)
        loss_tf.backward()
        opt_tf.step()

    # Evaluation
    bdh_acc = evaluate_graph_accuracy(model_bdh, val_samples, device=device)
    tf_acc = evaluate_graph_accuracy(model_tf, val_samples, device=device)

    print("\n" + "=" * 60)
    print(f"{'MODEL':<25} | {'GRAPH REACHABILITY ACCURACY':<25}")
    print("=" * 60)
    print(f"{'HatchlingLM (BDH)':<25} | {bdh_acc:23.1f}%")
    print(f"{'Standard Transformer':<25} | {tf_acc:23.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
