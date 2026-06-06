# Adversarial Attacks on ML Models — FGSM, PGD, and C&W on MNIST

> Comparative implementation and evaluation of three white-box adversarial attack methods against a CNN trained on MNIST — Fast Gradient Sign Method, Projected Gradient Descent, and Carlini & Wagner.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

---

## Overview

This project implements and compares three white-box adversarial attack methods against a convolutional neural network (CNN) trained to **99.1% clean accuracy** on the MNIST handwritten digit dataset. Each attack was implemented independently and evaluated under consistent metrics to produce a controlled, reproducible comparison.

**Central finding:** all three methods successfully fool a high-accuracy neural network using imperceptible perturbations, achieving fooling rates between 71.3% and 99.7%. The choice of attack method depends on the evaluation context — speed, perturbation efficiency, or rigour.

This was completed as **Assignment 3** for **Data Analytics in Cybersecurity (41180)** at the University of Technology Sydney.

---

## Results

| Attack | Config | Fooling Rate | Avg L2 Dist | Avg L-inf |
|---|---|---|---|---|
| FGSM | ε = 0.1 | 71.3% | 1.82 | 0.10 |
| FGSM | ε = 0.2 | 89.6% | 3.65 | 0.20 |
| FGSM | ε = 0.3 | 96.2% | 5.47 | 0.30 |
| PGD | ε = 0.1, α = 0.01, T = 40 | 95.1% | 1.78 | 0.10 |
| PGD | ε = 0.2, α = 0.01, T = 40 | 99.0% | 3.54 | 0.20 |
| **C&W (L2)** | c = 1e-4 | 98.4% | **0.94** | 0.09 |
| **C&W (L2)** | c = 1e-3 | **99.7%** | 1.31 | 0.12 |

**Key takeaways:**
- PGD outperforms FGSM at every equivalent ε — 95.1% vs 71.3% at ε = 0.1, with *lower* L2 distance
- C&W achieves the best fooling rate–distortion trade-off: 98.4% fooling at an average L2 of just 0.94
- C&W required ~50× more compute per sample than FGSM due to iterative Adam optimization and binary search over c

---

## Attack Methods

### 1. FGSM — Fast Gradient Sign Method
*Contributor: Senesh Chetan*

Single-step gradient-based attack. Moves the input in the direction of the loss gradient sign, scaled by ε:

```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

Requires only one forward + backward pass. Efficient but suboptimal — a linear approximation of the loss landscape.

```python
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.data.sign()
    adv_images = torch.clamp(images + perturbation, 0, 1)
    return adv_images.detach()
```

---

### 2. PGD — Projected Gradient Descent
*Contributor: Mandiv Ekanayake*

Iterative extension of FGSM. Repeats the gradient sign step T times with step size α, projecting back onto the ε-ball after each step. Random initialization explores different regions of perturbation space:

```
x^(t+1) = Clip_{x,ε} [ x^(t) + α · sign(∇_x J(θ, x^(t), y)) ]
```

Regarded as the standard white-box benchmark for adversarial robustness evaluation.

```python
def pgd_attack(model, images, labels, epsilon, alpha=None, num_iter=40):
    if alpha is None: alpha = epsilon / 10.0
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, 0, 1) - images
    delta.requires_grad_(True)
    for _ in range(num_iter):
        loss = F.cross_entropy(model(images + delta), labels)
        loss.backward()
        with torch.no_grad():
            delta.data += alpha * delta.grad.sign()
            delta.data  = delta.data.clamp(-epsilon, epsilon)
            delta.data  = (images + delta.data).clamp(0,1) - images
        delta.grad.zero_()
    return (images + delta).detach()
```

---

### 3. C&W — Carlini & Wagner (L2)
*Contributor: [Amri Zuhair](https://www.linkedin.com/in/amri-zuhair-227071334/)*

Optimization-based attack. Directly minimizes perturbation magnitude subject to misclassification via Adam optimizer. Uses a change of variables `δ = tanh(w) − x` to enforce box constraints, allowing unconstrained optimization. Binary search over `c` finds the minimum perturbation that still causes misclassification:

```
minimise  ||δ||_2  +  c · f(x + δ)
f(x') = max( max_{j≠t} Z(x')_j − Z(x')_t, −κ )
```

Consistently produces the smallest perturbations of any gradient-based attack.

```python
def cw_attack(model, images, labels, c=1e-3, kappa=0, num_steps=500, lr=0.01):
    w = torch.atanh(images.clamp(1e-6, 1-1e-6) * 2 - 1).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)
    for step in range(num_steps):
        x_adv  = (torch.tanh(w) + 1) / 2
        logits = model(x_adv)
        real   = logits[range(len(labels)), labels]
        other  = (logits - 1e9 * F.one_hot(labels, 10)).max(1).values
        f_val  = torch.clamp(real - other + kappa, min=0)
        l2     = ((x_adv - images)**2).sum(dim=(1,2,3)).sqrt().mean()
        loss   = l2 + c * f_val.mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return ((torch.tanh(w) + 1) / 2).detach()
```

---

## Model Architecture

CNN trained on MNIST to 99.1% clean accuracy:

| Layer | Details |
|---|---|
| Conv1 | 32 filters, 3×3 kernel, ReLU |
| Conv2 | 64 filters, 3×3 kernel, ReLU |
| FC1 | 128 units, ReLU |
| FC2 (Output) | 10 units, Softmax |

---

## Project Structure

```
adversarial-attacks-mnist/
│
├── fgsm_attack.ipynb          # FGSM implementation (Senesh Chetan)
├── pgd_attack.ipynb           # PGD implementation (Mandiv Ekanayake)
├── cw_attack.ipynb            # C&W implementation (Amri Zuhair)
│
└── models/
    └── mnist_cnn.pth          # Pretrained CNN weights
```

---

## Setup & Installation

```bash
git clone <repo-url>
cd adversarial-attacks-mnist
pip install torch torchvision matplotlib numpy
```

The MNIST dataset will download automatically via `torchvision.datasets.MNIST` on first run.

---

## Key Insights

- **FGSM's simplicity is both its strength and weakness.** A single gradient step is cheap and surprisingly effective, but it's a linear approximation — iterative methods find stronger adversarial examples within the same ε budget.
- **PGD random initialization matters.** PGD without random start consistently underperformed — initialization determines which region of the loss landscape is explored and avoids suboptimal local maxima.
- **C&W's `kappa` controls adversarial confidence.** `kappa=0` produces examples that barely cross the decision boundary; higher values produce wrong predictions with high confidence. Binary search over `c` is essential — fixed `c` leads to either failed attacks or unnecessarily large perturbations.
- **Adversarial vulnerability is not model-specific.** A 99.1%-accurate CNN is still systematically fooled by imperceptible perturbations. This has direct implications for ML-based cybersecurity systems such as intrusion detection and malware classifiers.

---

## Team

| Name |
|---|
| Senesh Chetan |
| Mandiv Ekanayake |
| [Amri Zuhair](https://www.linkedin.com/in/amri-zuhair-227071334/) |

---

## References

- Akhtar, N. et al. (2021). Advances in Adversarial Attacks and Defenses in Computer Vision. IEEE Access, 9, 155161–155196. https://doi.org/10.1109/ACCESS.2021.3127960
- Croce, F. & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. arXiv. https://arxiv.org/abs/2003.01690
- Bai, T. et al. (2021). Recent advances in adversarial training for adversarial robustness. IJCAI-21. https://doi.org/10.24963/ijcai.2021/591
- Machado, G.R. et al. (2023). Adversarial Machine Learning in Image Classification: A Survey. ACM Computing Surveys, 55(1). https://doi.org/10.1145/3485133
