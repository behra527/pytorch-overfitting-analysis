"""Evaluate the selected baseline checkpoint on the held-out test set."""

from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

try:
    from .data import prepare_data
    from .model import MoonClassifier
except ImportError:
    from data import prepare_data
    from model import MoonClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "baseline_best.pt"


def main() -> None:
    """Load the best baseline checkpoint and evaluate test performance."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Obtain the held-out test split.
    _, _, X_test, _, _, y_test, _ = prepare_data()

    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)

    # Recreate the baseline architecture.
    model = MoonClassifier(
        input_size=2,
        hidden_size=128,
        num_hidden_layers=3,
        dropout_rate=0.0,
    ).to(device)

    # Load saved weights safely as a state dictionary.
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(X_test_tensor)
        test_loss = criterion(logits, y_test_tensor).item()
        predictions = logits.argmax(dim=1)

    test_accuracy = (
        (predictions == y_test_tensor).float().mean().item()
    )

    y_true = y_test_tensor.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    print("\n=== Baseline Test-Set Evaluation ===")
    print(f"Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"Test samples: {len(y_test)}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Class 0", "Class 1"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
