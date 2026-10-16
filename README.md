

# Overfitting and Loss Curve Analysis with PyTorch



![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)

![Scikit-learn](https://img.shields.io/badge/scikit--learn-Data%20Processing-f7931e?logo=scikitlearn&logoColor=white)

![Status](https://img.shields.io/badge/Status-Completed-success)



A practical deep-learning project exploring overfitting, training and validation loss curves, and regularization techniques using PyTorch.



## Overview



A neural network may perform well on training data but generalize less effectively to unseen examples. This project investigates that behavior through training metrics, validation metrics, loss curves, and controlled experiments.



Several model configurations are compared to understand how early stopping, dropout, L2 regularization, and network size affect generalization.



## Objectives



\- Understand training loss versus validation loss.

\- Identify overfitting through learning curves.

\- Compare baseline training with regularization strategies.

\- Save the best model based on validation loss.

\- Evaluate the selected model on an independent test set.

\- Generate reproducible experiment summaries and visualizations.



## Tech Stack



\- Python

\- PyTorch

\- Scikit-learn

\- NumPy

\- Pandas

\- Matplotlib



## Dataset



The project generates 1,500 samples using Scikit-learn's `make_moons` function with noise set to `0.30`.



| Split | Samples | Percentage |

|---|---:|---:|

| Training | 1,050 | 70% |

| Validation | 225 | 15% |

| Test | 225 | 15% |

| Total | 1,500 | 100% |



Features are standardized using a `StandardScaler` fitted only on the training set to avoid data leakage.



## Model Architecture



The baseline is a multilayer perceptron (MLP) with:



\- 2 input features

\- 3 hidden layers

\- 128 units per hidden layer

\- ReLU activation

\- 2 output classes

\- Cross-entropy loss

\- Adam optimizer



The baseline learning rate is `0.001`, and training runs for up to 200 epochs.



## Experiments and Results



Five configurations were evaluated on the same data split.



| Experiment | Best Epoch | Best Validation Loss | Best Validation Accuracy |

|---|---:|---:|---:|

| Baseline | 43 | 0.2084 | 90.67% |

| Dropout (0.3) | 134 | 0.2108 | 90.67% |

| Early stopping | 13 | 0.2143 | 90.67% |

| L2 weight decay (0.01) | 93 | 0.2208 | 91.11% |

| Smaller network | 164 | 0.2214 | 89.33% |



The baseline achieved the lowest best validation loss among the tested configurations. Dropout produced a lower final validation loss than the baseline, while early stopping reduced training from 200 to 28 epochs.



These results apply to the tested configurations and one fixed split. They do not establish that a particular technique will always perform best.



## Visual Results



### Validation Loss Comparison



![Validation Loss Comparison](outputs/validation_loss_comparison.png)



### Training and Validation Accuracy



![Training and Validation Accuracy](outputs/accuracy_comparison.png)



## Held-Out Test Evaluation



The baseline checkpoint selected using validation loss was evaluated on the held-out test set.



| Metric | Result |

|---|---:|

| Test samples | 225 |

| Test loss | 0.1933 |

| Test accuracy | 92.44% |

| Macro F1-score | 92.44% |

| Correct predictions | 208 |

| Incorrect predictions | 17 |



### Confusion Matrix



Rows represent actual classes, and columns represent predicted classes.



```text

[[106   7]

&#x20;[ 10 102]]

```



The model correctly classified 208 of 225 test samples. This is an estimate from one held-out split, not a guarantee of performance on other datasets.



## Project Structure



```text

overfitting-loss-curves/

├── outputs/

│   ├── *_history.csv

│   ├── *_best.pt

│   ├── experiment_comparison.csv

│   ├── validation_loss_comparison.png

│   └── accuracy_comparison.png

├── reports/

│   └── final_report.md

├── src/

│   ├── data.py

│   ├── model.py

│   ├── train.py

│   ├── evaluate.py

│   └── compare_results.py

├── .gitignore

├── requirements.txt

└── README.md

```



## Setup



Clone the repository and navigate to the project directory.



Create and activate a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



## Usage



Run commands from the project root.



### Prepare and inspect the dataset



```powershell

python src/data.py

```



### Train the baseline model



```powershell

python -c "from src.train import run_experiment; run_experiment(epochs=200, hidden_size=128, num_hidden_layers=3, dropout_rate=0.0, weight_decay=0.0, patience=None, history_name='baseline_history.csv')"

```



### Run early stopping



```powershell

python src/train.py

```



### Compare experiment results



```powershell

python src/compare_results.py

```



### Evaluate the baseline checkpoint



```powershell

python src/evaluate.py

```



Training and evaluation generate experiment artifacts in the `outputs/` directory.



## Key Findings



\- Training loss alone is not sufficient to judge model generalization.

\- Validation loss can increase even while training continues to improve, indicating possible overfitting.

\- Early stopping can reduce unnecessary training epochs.

\- Dropout and L2 regularization can influence generalization differently.

\- The best configuration depends on the dataset and experimental settings.

\- A held-out test set provides a more independent estimate of performance than training metrics.



## Limitations



\- The dataset is synthetic and relatively small.

\- Experiments use one fixed random split.

\- Hyperparameters were not exhaustively tuned.

\- Results may differ on real-world datasets.

\- The selected model is not guaranteed to be optimal for every classification problem.



## Report



See [`reports/final_report.md`](reports/final_report.md) for the detailed experiment summary, test metrics, confusion matrix, conclusions, and limitations.





