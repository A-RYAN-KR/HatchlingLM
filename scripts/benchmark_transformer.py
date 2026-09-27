#!/usr/bin/env python3
"""
CLI script to benchmark HatchlingLM against Standard Causal Transformer baseline.
Compares parameter counts, training step throughput, loss convergence, and sparsity.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from data.dataloader import download_dataset, load_byte_data, get_batch
from hatchling.models.bdh import HatchlingLM
from hatchling.models.transformer import StandardTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BDH vs Transformer Baseline")
    parser.add_argument("--steps", type=int, default=500, help="Benchmark training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context length")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def benchmark_model(name, model, train_data, val_data, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    print(f"\n--- Benchmarking {name} ({args.steps} steps) ---")

    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(train_data, batch_size=args.batch_size, block_size=args.block_size, device=device)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    step_time_ms = (elapsed / args.steps) * 1000

    # Evaluate validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(20):
            vx, vy = get_batch(val_data, batch_size=args.batch_size, block_size=args.block_size, device=device)
            _, vloss = model(vx, targets=vy)
            val_losses.append(vloss.item())
    val_loss = sum(val_losses) / len(val_losses)

    print(f"{name} Completed in: {elapsed:.2f}s ({step_time_ms:.1f} ms/step) | Val Loss: {val_loss:.4f}")
    return {"elapsed": elapsed, "step_time_ms": step_time_ms, "val_loss": val_loss}


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on: {device}")

    data_path = download_dataset()
    train_data, val_data = load_byte_data(data_path)

    bdh = HatchlingLM(vocab_size=256, d_model=384, n_neurons=1536, n_layers=6, max_seq_len=args.block_size).to(device)
    tf = StandardTransformer(vocab_size=256, d_model=384, n_heads=6, d_ff=1536, n_layers=6, max_seq_len=args.block_size).to(device)

    p_bdh = sum(p.numel() for p in bdh.parameters())
    p_tf = sum(p.numel() for p in tf.parameters())

    print(f"HatchlingLM (BDH) Parameters:       {p_bdh:,} (~{p_bdh/1e6:.2f}M)")
    print(f"Standard Transformer Parameters:    {p_tf:,} (~{p_tf/1e6:.2f}M)")

    res_bdh = benchmark_model("HatchlingLM (BDH)", bdh, train_data, val_data, args, device)
    res_tf = benchmark_model("Standard Transformer", tf, train_data, val_data, args, device)

    print("\n" + "=" * 65)
    print("                     BENCHMARK SUMMARY")
    print("=" * 65)
    print(f" Metric                     HatchlingLM        Transformer")
    print("-" * 65)
    print(f" Total Parameters           {p_bdh:>11,}        {p_tf:>11,}")
    print(f" Total Time ({args.steps} steps)      {res_bdh['elapsed']:>9.2f}s        {res_tf['elapsed']:>9.2f}s")
    print(f" Step Latency               {res_bdh['step_time_ms']:>8.1f} ms        {res_tf['step_time_ms']:>8.1f} ms")
    print(f" Validation Loss            {res_bdh['val_loss']:>11.4f}        {res_tf['val_loss']:>11.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
