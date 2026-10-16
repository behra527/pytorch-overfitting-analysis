"""Dataset generation and splitting utilities."""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def prepare_data(random_state: int = 42):
    """Generate, scale, and split the nonlinear classification dataset."""

    X, y = make_moons(
        n_samples=1500,
        noise=0.30,
        random_state=random_state,
    )

    # 70% training, 15% validation, 15% testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    # Fit the scaler only on training data to prevent data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        y_train.astype(np.int64),
        y_val.astype(np.int64),
        y_test.astype(np.int64),
        scaler,
    )


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data()

    print("Dataset preparation successful.")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Feature count:      {X_train.shape[1]}")
    print(f"Training classes:   {np.bincount(y_train)}")
    print(f"Validation classes: {np.bincount(y_val)}")
    print(f"Test classes:       {np.bincount(y_test)}")
    print(f"Training dtype:     {X_train.dtype}")
    print(f"Label dtype:        {y_train.dtype}")
