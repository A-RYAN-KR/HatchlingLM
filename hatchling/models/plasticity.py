"""
Inference-time biological synaptic plasticity modules.
Implements:
1. Online Hebbian associative plasticity with decay (BDHPlasticBlock)
2. Stabilized homeostatic plasticity via Oja's normalization rule (BDHOjaBlock)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDHPlasticBlock(nn.Module):
    """
    Biological BDH Layer with Online Hebbian Synaptic Memory.
    Dynamically strengthens synaptic connections during streaming inference:
    W_t = decay * W_{t-1} + eta * (pre_synaptic (x) post_synaptic)
    """
    def __init__(
        self,
        d_model: int = 64,
        n_neurons: int = 128,
        sparsity_thresh: float = 0.05,
        eta: float = 0.08,
        decay: float = 0.98
    ):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.sparsity_thresh = sparsity_thresh
        self.eta = eta
        self.decay = decay

        # Structural learned weights
        self.w_in = nn.Linear(d_model, n_neurons, bias=False)
        self.w_out = nn.Linear(n_neurons, d_model, bias=False)

        # Base and dynamic synaptic memory
        self.register_buffer("base_synapse", torch.randn(n_neurons, n_neurons) * 0.02)
        self.dynamic_synapse = self.base_synapse.clone()

    def reset_synapses(self):
        """Reset fast synaptic memory to baseline structural state."""
        self.dynamic_synapse = self.base_synapse.clone()

    def reset(self):
        """Alias for reset_synapses."""
        self.reset_synapses()

    def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Process single token vector x_t: shape [1, d_model] or [1, 1, d_model].
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)

        # 1. Sparse non-negative neuron firing rate
        pre_act = self.w_in(x_t)
        act = F.relu(pre_act - self.sparsity_thresh)  # [1, n_neurons]

        # 2. Integrate lateral signal using current dynamic synaptic matrix
        lateral_signal = torch.matmul(act, self.dynamic_synapse.t())
        total_firing = act + F.relu(lateral_signal)  # [1, n_neurons]

        # 3. Dynamic Hebbian Update: dW = eta * (pre_synaptic^T @ post_synaptic)
        act_vec = act.squeeze(0)            # [n_neurons]
        total_vec = total_firing.squeeze(0)  # [n_neurons]
        hebbian_delta = self.eta * torch.outer(act_vec, total_vec)

        self.dynamic_synapse = (self.decay * self.dynamic_synapse) + hebbian_delta

        # 4. Project back to embedding space
        out = self.w_out(total_firing)
        return out + x_t


class BDHOjaBlock(nn.Module):
    """
    BDH Layer with stabilized homeostatic synaptic plasticity using Oja's Rule:
    dW = eta * (total * act^T - (total^2 * W))
    Prevents unbounded divergence while maintaining continuous associative plasticity.
    """
    def __init__(
        self,
        d_model: int = 64,
        n_neurons: int = 128,
        sparsity_thresh: float = 0.05,
        eta: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.sparsity_thresh = sparsity_thresh
        self.eta = eta

        self.w_in = nn.Linear(d_model, n_neurons, bias=False)
        self.w_out = nn.Linear(n_neurons, d_model, bias=False)
        self.register_buffer("base_synapse", torch.randn(n_neurons, n_neurons) * 0.02)
        self.dynamic_synapse = self.base_synapse.clone()

    def reset_synapses(self):
        """Reset fast synaptic memory to baseline structural state."""
        self.dynamic_synapse = self.base_synapse.clone()

    def reset(self):
        """Alias for reset_synapses."""
        self.reset_synapses()

    def forward_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Process single token vector x_t: shape [1, d_model] or [1, 1, d_model].
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)

        pre_act = self.w_in(x_t)
        act = F.relu(pre_act - self.sparsity_thresh).squeeze(0)  # [n_neurons]

        # Lateral recurrent signal
        lateral = torch.matmul(act, self.dynamic_synapse.t())
        total = act + F.relu(lateral)            # [n_neurons]

        # --- Oja's Homeostatic Update ---
        # dW = eta * (total * act^T - (total^2 * W))
        co_activation = torch.outer(total, act)
        homeostatic_drain = (total.unsqueeze(1) ** 2) * self.dynamic_synapse

        delta_w = self.eta * (co_activation - homeostatic_drain)
        self.dynamic_synapse = self.dynamic_synapse + delta_w

        out = self.w_out(total.unsqueeze(0))
        return out + x_t
