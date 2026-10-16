# Overfitting in Deep Networks and Reading Loss Curves

## 1. Objective

The objective of this project was to study overfitting in a deep neural
network, interpret training and validation loss curves, and compare
regularization techniques on a nonlinear binary classification task.

## 2. Dataset and Setup

- Dataset: Synthetic two-moons classification dataset
- Samples: 1,500
- Training split: 1,050 samples (70%)
- Validation split: 225 samples (15%)
- Test split: 225 samples (15%)
- Input features: 2
- Classes: 2
- Feature scaling: StandardScaler fitted only on training data
- Framework: PyTorch
- Hardware: CPU

## 3. Experiments

The following configurations were evaluated using validation data.

| Experiment | Best Epoch | Best Validation Loss | Best Validation Accuracy |
|---|---:|---:|---:|
| Baseline | 43 | 0.2084 | 90.67% |
| Dropout (0.3) | 134 | 0.2108 | 90.67% |
| Early stopping | 13 | 0.2143 | 90.67% |
| L2 weight decay (0.01) | 93 | 0.2208 | 91.11% |
| Small network | 164 | 0.2214 | 89.33% |

## 4. Overfitting Analysis

The baseline model achieved its lowest validation loss at epoch 43.
Its validation loss later increased to 0.3462 by epoch 200, while
training accuracy reached 92.00%. This divergence between training
and validation behavior indicates an overfitting pattern.

Dropout reduced the final validation loss to 0.2246 compared with
0.3462 for the baseline. Early stopping completed training in 28
epochs, reducing training time. In these experiments, neither L2
regularization nor the smaller network achieved a lower best
validation loss than the baseline.

These findings apply to the tested configurations and random split;
different hyperparameters or datasets may produce different results.

## 5. Final Test Evaluation

The baseline checkpoint selected using validation loss was evaluated
on the held-out test set.

- Test samples: 225
- Test loss: 0.1933
- Test accuracy: 92.44%
- Correct predictions: 208
- Incorrect predictions: 17
- Macro F1-score: 92.44%

### Classification Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Class 0 | 91.38% | 93.81% | 92.58% | 113 |
| Class 1 | 93.58% | 91.07% | 92.31% | 112 |

### Confusion Matrix

Rows represent actual classes; columns represent predicted classes.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 106 | 7 |
| Actual 1 | 10 | 102 |

## 6. Conclusion

The baseline achieved the lowest validation loss among the tested
configurations and 92.44% accuracy on the held-out test set.

The loss curves demonstrated that high training accuracy alone does
not guarantee improved generalization. Dropout and early stopping
helped control late-training deterioration in validation performance.

The experiments also demonstrated that regularization methods must
be evaluated empirically rather than assumed to improve every model.

## 7. Limitations

- The dataset is synthetic and relatively small.
- Results are based on one random split.
- Hyperparameters were not exhaustively tuned.
- Performance on real-world datasets may differ.
