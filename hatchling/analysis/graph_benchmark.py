"""
Algorithmic multi-step directed graph reachability benchmark.
Tests whether BDH and Transformer can deduce multi-hop relational paths without explicit knowledge graphs.
Matches Colab Notebook Extension 4 implementation.
"""

import random
from typing import Tuple, List, Union
import torch


def generate_graph_sample(num_nodes: int = 8, num_edges: int = 6, chain_len: int = 2) -> Tuple[str, int]:
    """
    Constructs a synthetic directed graph reachability test.

    Returns:
        (text, label) where label is 1 (reachable) or 0 (unreachable).
    """
    nodes = list(range(num_nodes))
    edges = set()

    # Create guaranteed directed path of specified length
    path = random.sample(nodes, chain_len + 1)
    for i in range(chain_len):
        edges.add((path[i], path[i + 1]))

    # Add random distractor edges
    while len(edges) < num_edges:
        u, v = random.sample(nodes, 2)
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))

    edges_list = list(edges)
    random.shuffle(edges_list)
    edge_str = ", ".join([f"{u}->{v}" for u, v in edges_list])

    # 50% positive reachability query, 50% negative query
    if random.random() < 0.5:
        start, end = path[0], path[-1]
        label = 1
    else:
        start, end = path[-1], path[0]
        label = 0

    text = f"Edges: {edge_str}. Path {start}->{end}? Answer: {label}"
    return text, label


def collate_graph_batch(
    samples: List[Tuple[str, int]],
    device: Union[str, torch.device] = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads and batches string graph queries into input and target byte tensors.
    """
    texts = [s[0].encode("utf-8") for s in samples]
    pad_len = max(len(t) for t in texts)
    batch_x = []
    batch_y = []

    for t in texts:
        arr = list(t)
        # Pad with 0
        arr = arr + [0] * (pad_len - len(arr))
        batch_x.append(arr[:-1])
        batch_y.append(arr[1:])

    bx = torch.tensor(batch_x, dtype=torch.long, device=device)
    by = torch.tensor(batch_y, dtype=torch.long, device=device)
    return bx, by


@torch.no_grad()
def evaluate_graph_accuracy(
    model: torch.nn.Module,
    samples: List[Tuple[str, int]],
    device: Union[str, torch.device] = "cuda"
) -> float:
    """
    Evaluates exact match accuracy on the final '0' or '1' token.
    """
    model.eval()
    correct = 0

    for text, label in samples:
        prompt = text.split("Answer: ")[0] + "Answer: "
        idx = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)

        logits, _ = model(idx)
        pred_byte = torch.argmax(logits[:, -1, :], dim=-1).item()
        pred_char = chr(pred_byte) if 0 <= pred_byte < 128 else ""

        if pred_char == str(label):
            correct += 1

    return (correct / len(samples)) * 100.0 if len(samples) > 0 else 0.0
