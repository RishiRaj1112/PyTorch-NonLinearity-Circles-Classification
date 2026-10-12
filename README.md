# PyTorch Non-Linear Classification & SVM Benchmark (Circles)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishiRaj1112/PyTorch-NonLinearity-Circles-Classification/blob/main/PyTorch_Non_Linearity_Classification.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end investigation into **non-linear decision boundaries in neural networks**, demonstrating why linear models fail on concentric circles and how **ReLU activation functions** and **Support Vector Machines (RBF Kernel)** solve non-linear classification.

---

## 📌 Problem Overview

When training machine learning models on non-linearly separable datasets (such as concentric circles from `make_circles`), a purely linear model cannot draw a curved decision boundary to separate the inner circle from the outer ring.

This repository demonstrates:
1. **Model 0 (Linear Network):** A multi-layer neural network with no activation functions collapses mathematically into a single linear model, capping out at **~50% accuracy**.
2. **Model 1 (Non-Linear Network with ReLU):** By introducing `nn.ReLU()` non-linearities, the network learns piecewise-linear decision boundaries that encircle the inner class, achieving **100% test accuracy**.
3. **Classical ML Benchmark (SVM with RBF Kernel):** A Support Vector Classifier projecting features via the Radial Basis Function kernel achieves **100% accuracy**.

---

## 📊 Benchmark Summary

| Model | Architecture / Method | Test Accuracy | Decision Boundary Behavior |
| :--- | :--- | :--- | :--- |
| **PyTorch Model 0 (Linear Only)** | 3 Linear layers (no activation) | **50.00%** | Straight line hyperplane; completely unable to separate concentric rings |
| **PyTorch Model 1 (Non-Linear)** | 3 Linear layers + 2 `nn.ReLU()` | **100.00%** | Piecewise-linear boundary curving around the inner circle |
| **Support Vector Machine (SVM)** | Scikit-Learn `SVC(kernel='rbf')` | **100.00%** | Smooth circular contour via infinite-dimensional feature mapping |

---

## 🧠 Architectures

### Model 0: Purely Linear
```python
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
```

### Model 1: Non-Linear with ReLU
```python
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
```

---

## 🚀 How to Run

### Option 1: Open in Google Colab (One-Click)
Click the badge below to run the fully executed notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishiRaj1112/PyTorch-NonLinearity-Circles-Classification/blob/main/PyTorch_Non_Linearity_Classification.ipynb)

### Option 2: Local Installation

```bash
git clone https://github.com/RishiRaj1112/PyTorch-NonLinearity-Circles-Classification.git
cd PyTorch-NonLinearity-Circles-Classification
pip install -r requirements.txt
jupyter notebook PyTorch_Non_Linearity_Classification.ipynb
```

---

## 🙏 Acknowledgments

* Foundational workflow, `make_circles` formulation, and decision boundary helper functions adapted from **Daniel Bourke's** [PyTorch Deep Learning Course](https://github.com/mrdbourke/pytorch-deep-learning) (FreeCodeCamp).
* Extended with classical SVM (RBF Kernel) benchmarks and comparative analysis.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
