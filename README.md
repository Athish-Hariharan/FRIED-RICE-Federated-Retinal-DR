# FRIED RICE
## Federated Retinal Image Evaluation & Diagnosis via Resilient Inter-Client Edge Learning

---

## Overview

FRIED RICE is a deep learning framework for diabetic retinopathy grading using fundus retinal images. The project investigates both centralized and federated learning paradigms under class imbalance and non-IID data conditions.

---

## Dataset

- APTOS 2019 Diabetic Retinopathy Dataset
- 5-class severity grading (0–4)
- Highly imbalanced class distribution

---

## Centralized Baseline (ResNet18 + Weighted Loss)

### Model
- ResNet18 (ImageNet pretrained)
- Final fully connected layer modified for 5 classes

### Training Setup
- Loss: Weighted CrossEntropy
- Optimizer: SGD (lr=0.001, momentum=0.9)
- Batch size: 16
- Epochs: 30
- Image size: 224×224

### Performance

- Best Validation Accuracy: **85.79%**
- Macro F1 Score: **0.73**

### Observations

- Transfer learning significantly improves minority recall.
- Weighted loss reduces imbalance bias.
- Deep architectures are critical for fine-grained DR grading.

---

## Federated Learning Motivation

Medical data is often distributed across hospitals and cannot be centralized due to privacy constraints. This project simulates federated learning under extreme non-IID data partitions to evaluate performance degradation and convergence behavior.

---

## Repository Structure

data/
models/
training/
federated/
docs/

---

## Experimental Evolution

| Version | Description |
|----------|------------|
| v1 | CNN baseline |
| v2 | ResNet18 centralized |
| v3 | FedAvg naive |
| v4 | FedAvg stabilized |

---

## Future Work

- Partial non-IID experiments
- FedProx implementation
- Client-level weighting strategies
- Minority class stabilization

---
