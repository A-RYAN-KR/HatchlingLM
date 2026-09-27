"""
Byte-level dataset downloading and batch loading for HatchlingLM.
Uses raw UTF-8 bytes (0-255 vocabulary) with zero subword tokenizer dependencies.
"""

import os
import urllib.request
from typing import Tuple
import numpy as np
import torch

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_dataset(data_dir: str = "./data", url: str = DEFAULT_DATA_URL) -> str:
    """
    Downloads Tiny Shakespeare if not already present.

    Args:
        data_dir: Directory where the dataset file should reside.
        url: Remote URL of the text corpus.

    Returns:
        Absolute or relative path to the downloaded file.
    """
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, data_path)
        print(f"Download complete: {data_path}")
    return data_path


def load_byte_data(
    data_path: str = "./data/tinyshakespeare.txt",
    split_ratio: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads raw text into uint8 byte-level token tensors.

    Args:
        data_path: Path to raw text file.
        split_ratio: Train/Validation split fraction.

    Returns:
        (train_data, val_data) as 1D PyTorch LongTensors.
    """
    if not os.path.exists(data_path):
        data_path = download_dataset(os.path.dirname(data_path) or "./data")

    with open(data_path, "rb") as f:
        raw_bytes = f.read()

    tokens = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8).copy()).long()
    split_idx = int(split_ratio * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]
    return train_data, val_data


def get_batch(
    data: torch.Tensor,
    batch_size: int = 16,
    block_size: int = 128,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a random batch of byte sequences (x) and next-byte targets (y).

    Args:
        data: 1D PyTorch tensor of token IDs.
        batch_size: Batch dimension B.
        block_size: Context length T.
        device: Target device string or torch.device.

    Returns:
        (x, y) tensors of shape [B, T].
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
