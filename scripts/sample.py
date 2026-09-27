#!/usr/bin/env python3
"""
CLI script to sample autoregressive text generation from HatchlingLM.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from hatchling.models.bdh import HatchlingLM
from hatchling.engine.generate import generate


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from HatchlingLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hatchling_best.pt", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="KING LEAR:\n", help="Prompt string")
    parser.add_argument("--max_tokens", type=int, default=300, help="Number of bytes to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = HatchlingLM(vocab_size=256, d_model=384, n_neurons=1536, n_layers=6, max_seq_len=128).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Notice: No checkpoint found at '{args.checkpoint}'. Generating from initialized weights.")

    print(f"\nPrompt: {repr(args.prompt)}")
    print(f"Temperature: {args.temperature} | Top-K: {args.top_k}")
    print("-" * 60)
    output = generate(
        model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()
