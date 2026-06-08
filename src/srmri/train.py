"""Loop di training per la rete di super-risoluzione.

Loss di default: L1 (più robusta percettivamente della MSE per la SR; cfr.
pratica comune post-SRCNN). Supporta anche MSE. Restituisce lo storico delle
loss per poter verificare la convergenza.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device(prefer: str | None = None) -> torch.device:
    """Auto-detect del device (cuda se disponibile, altrimenti cpu)."""
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loss_fn(name: str) -> nn.Module:
    name = name.lower()
    if name in ("l1", "mae"):
        return nn.L1Loss()
    if name in ("l2", "mse"):
        return nn.MSELoss()
    raise ValueError(f"Loss sconosciuta: {name!r}")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Esegue un'epoca; ritorna la loss media."""
    model.train()
    total, n = 0.0, 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        pred = model(lr)
        loss = criterion(pred, hr)
        loss.backward()
        optimizer.step()
        total += loss.item() * lr.size(0)
        n += lr.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, loader: Iterable, criterion: nn.Module, device: torch.device
) -> float:
    """Loss media su un loader (senza grad)."""
    model.eval()
    total, n = 0.0, 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        loss = criterion(pred, hr)
        total += loss.item() * lr.size(0)
        n += lr.size(0)
    return total / max(n, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    loss: str = "l1",
    device: torch.device | None = None,
    val_loader: DataLoader | None = None,
    verbose: bool = False,
) -> dict:
    """Addestra il modello; ritorna dict con storico loss e device usato."""
    device = device or get_device()
    model.to(device)
    criterion = _loss_fn(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history["train_loss"].append(tr)
        if val_loader is not None:
            va = evaluate_loss(model, val_loader, criterion, device)
            history["val_loss"].append(va)
        if verbose:
            msg = f"epoch {epoch + 1}/{epochs} - train_loss={tr:.5f}"
            if val_loader is not None:
                msg += f" - val_loss={history['val_loss'][-1]:.5f}"
            print(msg)
    return {"history": history, "device": str(device)}
