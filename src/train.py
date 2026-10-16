"""Train a neural network and record training/validation losses."""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .data import prepare_data
    from .model import MoonClassifier
except ImportError:
    from data import prepare_data
    from model import MoonClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_experiment(
    epochs: int = 200,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    hidden_size: int = 128,
    num_hidden_layers: int = 3,
    dropout_rate: float = 0.0,
    weight_decay: float = 0.0,
    patience: int | None = None,
    history_name: str = "baseline_history.csv",
) -> pd.DataFrame:
    """Train the model and optionally stop when validation loss plateaus."""

    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, _, y_train, y_val, _, _ = prepare_data()
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    model = MoonClassifier(
        input_size=2,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            count = labels.size(0)
            train_loss_sum += loss.item() * count
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += count

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for features, labels in val_loader:
                logits = model(features)
                loss = criterion(logits, labels)

                count = labels.size(0)
                val_loss_sum += loss.item() * count
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_count += count

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_correct / train_count,
                "val_accuracy": val_correct / val_count,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Train acc: {train_correct / train_count:.3f} | "
                f"Val acc: {val_correct / val_count:.3f}"
            )

        if patience is not None and epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best validation loss: {best_val_loss:.4f}"
            )
            break

    model.load_state_dict(best_weights)

    history_df = pd.DataFrame(history)
    history_path = OUTPUT_DIR / history_name
    history_df.to_csv(history_path, index=False)

    checkpoint_path = OUTPUT_DIR / history_name.replace("_history.csv", "_best.pt")
    torch.save(model.state_dict(), checkpoint_path)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Epochs completed: {len(history_df)}")
    print(f"History saved: {history_path}")
    print(f"Best model saved: {checkpoint_path}")

    return history_df


def main() -> None:
    """Run the baseline experiment with early stopping enabled."""
    run_experiment(
        epochs=200,
        patience=15,
        history_name="early_stopping_history.csv",
    )


if __name__ == "__main__":
    main()

