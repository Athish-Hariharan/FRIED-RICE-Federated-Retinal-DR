---

# 🍚 FRIED RICE

## Federated Retinal Image Evaluation & Diagnosis via Resilient Inter-Client Edge-learning

---

## 🧠 Overview

FRIED RICE is a research-oriented deep learning framework for **Diabetic Retinopathy (DR) grading** using fundus retinal images.

This project systematically investigates:

* Centralized training under class imbalance
* Federated Learning under statistical heterogeneity
* Impact of non-IID client distributions
* Stabilization via FedProx
* Convergence degradation under extreme label skew

The objective is to empirically quantify how federated learning behaves under realistic medical data distribution scenarios.

---

## 📊 Dataset

**APTOS 2019 Diabetic Retinopathy Dataset**

* 5 severity classes (0–4)
* 2,930 training samples
* Severe class imbalance

| Class | Samples |
| ----- | ------- |
| 0     | 1434    |
| 1     | 300     |
| 2     | 808     |
| 3     | 154     |
| 4     | 234     |

Weighted loss is used to mitigate imbalance bias.

---

# 🏥 Centralized Baseline

## Model

* ResNet18 (ImageNet pretrained)
* Final fully connected layer modified for 5-class classification

## Training Setup

| Parameter     | Value                 |
| ------------- | --------------------- |
| Loss          | Weighted CrossEntropy |
| Optimizer     | SGD                   |
| Learning Rate | 0.001                 |
| Momentum      | 0.9                   |
| Batch Size    | 16                    |
| Epochs        | 30                    |
| Image Size    | 224×224               |

## Performance

* **Best Validation Accuracy: 85.79%**
* Macro F1 Score: ~0.73

### Observations

* Transfer learning significantly improves convergence.
* Weighted loss improves minority-class recall.
* Deep pretrained backbones are critical for DR severity grading.

---

# 🌐 Federated Learning Study

## Motivation

Medical imaging data is:

* Distributed across institutions
* Privacy-restricted
* Demographically heterogeneous

Federated Learning enables collaborative training without centralizing data.

All experiments simulate **3 clients**.

---

# 📉 Heterogeneity Simulation

Data is partitioned using a Dirichlet distribution:

* α = 1.0 → Near IID
* α = 0.5 → Moderate heterogeneity
* α = 0.1 → Severe non-IID

Lower α induces stronger label skew per client.

---

# 📊 Federated Results (3 Clients)

## FedAvg vs FedProx (μ = 0.01)

| α   | FedAvg Best Acc | FedProx Best Acc |
| --- | --------------- | ---------------- |
| 1.0 | ~82%            | ~83%             |
| 0.5 | ~77%            | ~78%             |
| 0.1 | ~62%            | ~62%             |

---

## 🔎 Key Observations

### 1️⃣ Heterogeneity Collapse

As α decreases:

* Client divergence increases
* Convergence slows
* Global accuracy degrades

Severe heterogeneity (α=0.1) produces ~23% degradation relative to centralized training.

---

### 2️⃣ FedProx Stabilization

FedProx:

* Improves stability under mild/moderate heterogeneity
* Slightly improves best accuracy for α=1.0 and α=0.5
* Does not recover performance under extreme skew (α=0.1)

This behavior aligns with known theoretical limitations of proximal regularization under strong statistical heterogeneity.

---

### 3️⃣ Centralized vs Federated Gap

| Setting      | Best Accuracy |
| ------------ | ------------- |
| Centralized  | 85.79%        |
| FedAvg α=1.0 | ~82%          |
| FedAvg α=0.1 | ~62%          |

The performance gap increases monotonically with statistical heterogeneity.

---

# 🧪 Experimental Logging

Results automatically stored in:

```
logs/results/
```

Plots generated in:

```
logs/plots/
```

Includes:

* Accuracy vs Round
* Accuracy vs Alpha
* JSON experiment logs
* Automated comparison utilities

---

# 📁 Repository Structure

```
data/
models/
training/
federated/
experiments/
logs/
docs/
```

---

# 🔄 Version History

| Tag                     | Description                       |
| ----------------------- | --------------------------------- |
| v1_cnn_weighted         | Custom CNN baseline               |
| v2_resnet_weighted      | Centralized ResNet18              |
| v3_fedavg_shard         | Shard-based extreme non-IID       |
| v4_dirichlet_simulation | Dirichlet-based FL                |
| v5_fedprox_study        | FedProx stabilization experiments |

---

# 🎓 Contributions

This repository provides:

* Empirical quantification of heterogeneity-induced degradation
* Dirichlet-based non-IID simulation framework
* FedAvg vs FedProx comparative analysis
* Reproducible logging and plotting pipeline
* Modular FL experimentation platform

---

# 🚀 Future Directions

* Increase number of clients (3 → 10+)
* Adaptive client weighting
* Backbone freezing during FL
* Personalized FL
* Minority-aware aggregation
* Domain shift simulation across hospitals

---

# 🏁 Reproducibility

Centralized training:

```
python -m training.train
```

Federated training:

```
python -m training.federated_train --method fedavg --alpha 0.5 --mu 0.0
```

Plot comparison:

```
python -m training.plot_alpha_comparison
```

---

# 📌 Conclusion

FRIED RICE demonstrates that:

* Federated learning performance strongly depends on client data heterogeneity.
* Mild non-IID maintains competitive performance.
* Severe heterogeneity causes substantial degradation.
* FedProx provides limited stabilization but does not fully resolve divergence.

This framework serves as a foundation for advanced federated medical AI research.

---
