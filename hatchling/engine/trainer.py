"""
Training engine for HatchlingLM with AMP, cosine decay with warmup,
and resumable checkpointing.
"""

import math
import os
import time
from typing import Dict, Optional, Callable
import torch
from tqdm.auto import tqdm


def get_lr_scheduler(
    step: int,
    warmup_steps: int = 150,
    max_steps: int = 2000,
    learning_rate: float = 6e-4,
    min_lr: float = 6e-5
) -> float:
    """
    Computes learning rate with linear warmup and cosine decay.
    """
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    get_batch_fn: Callable[[str, int, int, str], tuple],
    eval_iters: int = 30,
    batch_size: int = 32,
    block_size: int = 128,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluates train and validation loss over multiple Monte Carlo batches.
    """
    out = {}
    model.eval()
    is_cuda = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch_fn(split, batch_size=batch_size, block_size=block_size, device=device)
            if is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, targets=y)
            else:
                _, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class Trainer:
    """
    High-level training coordinator for HatchlingLM and baselines.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        learning_rate: float = 6e-4,
        min_lr: float = 6e-5,
        warmup_steps: int = 150,
        max_steps: int = 2000,
        batch_size: int = 32,
        block_size: int = 128,
        eval_interval: int = 100,
        eval_iters: int = 30,
        checkpoint_dir: str = "checkpoints",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data

        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        self.is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.is_cuda else None
        self.history = {"train_loss": [], "val_loss": [], "steps": []}

    def _sample_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)

    def train(self) -> Dict:
        self.model.train()
        pbar = tqdm(range(self.max_steps), desc="Training HatchlingLM")
        best_val_loss = float("inf")
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "hatchling_best.pt")

        for step in pbar:
            lr = get_lr_scheduler(
                step,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                min_lr=self.min_lr
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            xb, yb = self._sample_batch("train")

            if self.is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, loss = self.model(xb, targets=yb)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, loss = self.model(xb, targets=yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if step % self.eval_interval == 0 or step == self.max_steps - 1:
                losses = estimate_loss(
                    self.model,
                    lambda split, batch_size, block_size, device: self._sample_batch(split),
                    eval_iters=self.eval_iters,
                    batch_size=self.batch_size,
                    block_size=self.block_size,
                    device=self.device
                )
                train_loss, val_loss = losses["train"], losses["val"]
                self.history["steps"].append(step)
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                pbar.set_postfix({"train": f"{train_loss:.3f}", "val": f"{val_loss:.3f}"})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "step": step,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": best_val_loss,
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, best_checkpoint_path)

        return self.history
