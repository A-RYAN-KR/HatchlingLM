"""
Unit tests for synaptic plasticity models (Hebbian and Oja's rule).
"""

import torch
import pytest
from hatchling.models.plasticity import BDHPlasticBlock, BDHOjaBlock


def test_hebbian_plasticity_step():
    d_model, n_neurons = 32, 64
    plastic_block = BDHPlasticBlock(d_model=d_model, n_neurons=n_neurons, eta=0.05, decay=0.98)

    w_before = plastic_block.dynamic_synapse.clone()
    x_t = torch.randn(1, d_model)

    out = plastic_block.forward_step(x_t)
    assert out.shape == (1, d_model)

    w_after = plastic_block.dynamic_synapse
    # Synapse must have updated due to token processing
    drift = torch.norm(w_after - w_before).item()
    assert drift > 0.0, "Hebbian synapse must dynamically adapt on-the-fly"

    # Test reset
    plastic_block.reset_synapses()
    assert torch.allclose(plastic_block.dynamic_synapse, plastic_block.base_synapse)


def test_oja_plasticity_step():
    d_model, n_neurons = 32, 64
    oja_block = BDHOjaBlock(d_model=d_model, n_neurons=n_neurons, eta=0.01)

    w_before = oja_block.dynamic_synapse.clone()
    x_t = torch.randn(1, d_model)

    out = oja_block.forward_step(x_t)
    assert out.shape == (1, d_model)

    w_after = oja_block.dynamic_synapse
    drift = torch.norm(w_after - w_before).item()
    assert drift > 0.0, "Oja synapse must update with token step"
