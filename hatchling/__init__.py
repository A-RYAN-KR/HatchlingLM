"""
HatchlingLM: Dragon Hatchling Architecture
A biologically grounded neural language model with sparse non-negative activations,
lateral synaptic recurrence, and inference-time synaptic plasticity.
"""

from hatchling.models.bdh import BDHBlock, HatchlingLM
from hatchling.models.transformer import StandardTransformer

__version__ = "0.1.0"
__all__ = ["BDHBlock", "HatchlingLM", "StandardTransformer"]
