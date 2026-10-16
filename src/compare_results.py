
"""Compare experiment histories and plot loss and accuracy curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPERIMENTS = {
    "Baseline": "baseline_history.csv",
    "Early stopping": "early_stopping_history.csv",
    "Dropout 0.3": "dropout_history.csv",
    "Small network": "small_network_history.csv",
    "L2 weight decay 0.01": "l2_history.csv",
}


def main() -> None:
    """Summarize experiment histories and generate comparison plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    histories = {}

    for name, filename in EXPERIMENTS.items():
        path = OUTPUT_DIR / filename

        if not path.exists():
            print(f"Skipping missing history: {path}")
            continue

        history = pd.read_csv(path)

        required_columns = {
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
        }

        missing_columns = required_columns - set(history.columns)
        if missing_columns:
            print(
                f"Skipping {name}: missing columns "
                f"{sorted(missing_columns)}"
            )
            continue

        if history.empty:
            print(f"Skipping empty history: {path}")
            continue

        histories[name] = history

        # Find the epoch with the lowest validation loss.
        best_idx = history["val_loss"].idxmin()
        best = history.loc[best_idx]
        final = history.iloc[-1]

        summaries.append(
            {
                "experiment": name,
                "epochs_completed": len(history),
                "best_epoch": int(best["epoch"]),
                "best_train_loss": best["train_loss"],
                "best_val_loss": best["val_loss"],
                "best_train_accuracy": best["train_accuracy"],
                "best_val_accuracy": best["val_accuracy"],
                "final_train_loss": final["train_loss"],
                "final_val_loss": final["val_loss"],
                "final_train_accuracy": final["train_accuracy"],
                "final_val_accuracy": final["val_accuracy"],
            }
        )

    if not summaries:
        raise RuntimeError(
            "No valid experiment history CSV files were found."
        )

    # Save the full comparison report, ranked by best validation loss.
    report = pd.DataFrame(summaries).sort_values(
        "best_val_loss",
        ascending=True,
    )

    report_path = OUTPUT_DIR / "experiment_comparison.csv"
    report.to_csv(report_path, index=False)

    print("\n=== Experiment Comparison ===")
    print("Sorted by lowest validation loss; accuracy values are percentages.")

    display_report = report[
        [
            "experiment",
            "epochs_completed",
            "best_epoch",
            "best_val_loss",
            "best_val_accuracy",
            "final_train_accuracy",
            "final_val_accuracy",
            "final_val_loss",
        ]
    ].copy()

    # Convert accuracy fractions to percentages for easier reading.
    accuracy_columns = [
        "best_val_accuracy",
        "final_train_accuracy",
        "final_val_accuracy",
    ]
    display_report[accuracy_columns] *= 100

    print(display_report.round(3).to_string(index=False))
    print(f"\nComparison CSV saved: {report_path}")

    # Plot validation loss curves.
    plt.figure(figsize=(11, 6))

    for name, history in histories.items():
        plt.plot(
            history["epoch"],
            history["val_loss"],
            label=name,
            linewidth=1.8,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross-Entropy Loss")
    plt.title("Validation Loss Comparison Across Experiments")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = OUTPUT_DIR / "validation_loss_comparison.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print(f"Validation loss plot saved: {loss_path}")

    # Plot training and validation accuracy side by side.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, history in histories.items():
        axes[0].plot(
            history["epoch"],
            history["train_accuracy"] * 100,
            label=name,
            linewidth=1.6,
        )

        axes[1].plot(
            history["epoch"],
            history["val_accuracy"] * 100,
            label=name,
            linewidth=1.6,
        )

    axes[0].set_title("Training Accuracy Comparison")
    axes[1].set_title("Validation Accuracy Comparison")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)

    fig.tight_layout()

    accuracy_path = OUTPUT_DIR / "accuracy_comparison.png"
    fig.savefig(accuracy_path, dpi=150)
    plt.close(fig)

    print(f"Accuracy comparison plot saved: {accuracy_path}")


if __name__ == "__main__":
    main()
