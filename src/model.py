"""Configurable neural network for overfitting experiments."""

import torch
from torch import nn


class MoonClassifier(nn.Module):
    """MLP classifier with configurable width, depth, and dropout."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_hidden_layers: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1.")

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")

        layers = []
        in_features = input_size

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            in_features = hidden_size

        # Raw logits are returned; CrossEntropyLoss handles class probabilities.
        layers.append(nn.Linear(in_features, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for the two classes."""
        return self.network(x)


if __name__ == "__main__":
    model = MoonClassifier()
    sample = torch.randn(8, 2)
    output = model(sample)

    print(model)
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
