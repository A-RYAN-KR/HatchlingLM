"""
Core BDH (Dragon Hatchling) neural architecture implementation.
Features sparse non-negative neuron activations, afferent/lateral/efferent synaptic flow,
and residual connections.
"""

from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BDHBlock(nn.Module):
    """
    Core BDH Layer: Sparse non-negative neuron activation
    with biologically inspired lateral synaptic integration.
    """
    def __init__(self, d_model: int, n_neurons: int, sparsity_thresh: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.sparsity_thresh = sparsity_thresh

        self.ln = nn.LayerNorm(d_model)
        self.w_in = nn.Linear(d_model, n_neurons, bias=False)
        self.w_recurrent = nn.Linear(n_neurons, n_neurons, bias=False)
        self.w_out = nn.Linear(n_neurons, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the BDH block.

        Args:
            x: Input tensor of shape [B, T, d_model].
            return_activations: If True, also returns the non-negative neuron activations.

        Returns:
            Output tensor of shape [B, T, d_model], or (out, act) if return_activations=True.
        """
        residual = x
        x_norm = self.ln(x)
        pre_act = self.w_in(x_norm)  # [B, T, n_neurons]
        # Strictly non-negative biological firing via thresholded ReLU
        act = F.relu(pre_act - self.sparsity_thresh)

        # Lateral synaptic communication
        synaptic_signal = self.w_recurrent(act)
        total_firing = act + F.relu(synaptic_signal)

        # Efferent projection
        out = residual + self.w_out(total_firing)

        if return_activations:
            return out, act
        return out


class HatchlingLM(nn.Module):
    """
    HatchlingLM: Biologically Grounded Language Model
    Replaces dense Transformer self-attention with sparse synaptic cortical columns.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 384,
        n_neurons: int = 1536,
        n_layers: int = 6,
        max_seq_len: int = 128,
        sparsity_thresh: float = 0.05
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.sparsity_thresh = sparsity_thresh

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            BDHBlock(d_model, n_neurons, sparsity_thresh=sparsity_thresh)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: token embedding and output head share weights
        self.tok_emb.weight = self.head.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_activations: bool = False
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]
    ]:
        """
        Forward pass for HatchlingLM.

        Args:
            idx: LongTensor of input byte token IDs [B, T].
            targets: Optional LongTensor of target byte token IDs [B, T].
            return_activations: Whether to collect and return per-layer neuron firing rates.

        Returns:
            (logits, loss) or (logits, loss, layer_activations)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        layer_activations = []
        for layer in self.layers:
            if return_activations:
                x, acts = layer(x, return_activations=True)
                layer_activations.append(acts)
            else:
                x = layer(x)

        x = self.final_ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        if return_activations:
            return logits, loss, layer_activations
        return logits, loss
