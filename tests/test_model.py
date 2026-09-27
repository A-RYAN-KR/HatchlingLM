"""
Unit tests for BDH model components and HatchlingLM.
"""

import torch
import pytest
from hatchling.models.bdh import BDHBlock, HatchlingLM
from hatchling.models.transformer import StandardTransformer, CausalSelfAttention


def test_bdh_block_shape():
    batch_size, seq_len, d_model, n_neurons = 2, 16, 64, 128
    block = BDHBlock(d_model=d_model, n_neurons=n_neurons, sparsity_thresh=0.05)
    x = torch.randn(batch_size, seq_len, d_model)

    out = block(x)
    assert out.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"

    out, act = block(x, return_activations=True)
    assert act.shape == (batch_size, seq_len, n_neurons), f"Expected activations shape {(batch_size, seq_len, n_neurons)}, got {act.shape}"
    # Sparsity condition: all activations must be non-negative
    assert (act >= 0.0).all(), "BDH activations must be strictly non-negative (biological property)"


def test_hatchling_lm_forward_and_loss():
    vocab_size, d_model, n_neurons, n_layers, seq_len = 256, 64, 128, 2, 32
    model = HatchlingLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_neurons=n_neurons,
        n_layers=n_layers,
        max_seq_len=64
    )

    idx = torch.randint(0, vocab_size, (2, seq_len))
    targets = torch.randint(0, vocab_size, (2, seq_len))

    # Test forward with loss
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, seq_len, vocab_size)
    assert loss is not None and loss.item() > 0.0

    # Test forward with activations
    logits, loss, acts = model(idx, targets=targets, return_activations=True)
    assert len(acts) == n_layers
    assert acts[0].shape == (2, seq_len, n_neurons)


def test_standard_transformer_forward():
    vocab_size, d_model, seq_len = 256, 64, 16
    model = StandardTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, d_ff=128, n_layers=2)
    idx = torch.randint(0, vocab_size, (2, seq_len))
    targets = torch.randint(0, vocab_size, (2, seq_len))

    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, seq_len, vocab_size)
    assert loss is not None
