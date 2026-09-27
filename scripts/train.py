#!/usr/bin/env python3
"""
CLI training script for HatchlingLM.
Supports cosine learning rate scheduling, Automatic Mixed Precision (AMP),
and automatic best-checkpoint persistence.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from data.dataloader import download_dataset, load_byte_data
from hatchling.models.bdh import HatchlingLM
from hatchling.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train HatchlingLM (BDH Architecture)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context length")
    parser.add_argument("--max_steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Linear warmup steps")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=30, help="Batches per evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training HatchlingLM on device: {device}")

    # Prepare data
    data_path = download_dataset()
    train_data, val_data = load_byte_data(data_path)
    print(f"Dataset loaded: {len(train_data):,} train bytes, {len(val_data):,} val bytes")

    # Instantiate model
    model = HatchlingLM(
        vocab_size=256,
        d_model=384,
        n_neurons=1536,
        n_layers=6,
        max_seq_len=args.block_size
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")

    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )

    history = trainer.train()

    # Save loss curve
    if len(history["steps"]) > 0:
        os.makedirs("assets", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(history["steps"], history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(history["steps"], history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
        plt.title("HatchlingLM: Training Dynamics", fontsize=12, fontweight="bold")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("assets/loss_curve.png", dpi=150)
        plt.close()
        print("Loss curve saved to assets/loss_curve.png")


if __name__ == "__main__":
    main()
