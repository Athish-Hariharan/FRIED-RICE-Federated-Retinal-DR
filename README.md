# 🍚 FRIED-RICE

### Federated Research Integrated Environment for Diabetic Retinopathy

---

## 🧠 Project Evolution 

### 🔹 Phase 1 — Core Research (Baseline System)

Initially, this project focused on **federated learning for diabetic retinopathy classification**:

* Centralized ResNet18 baseline
* Federated learning using:

  * FedAvg
  * FedProx
* Dirichlet-based non-IID simulation (α = 0.1 → 1.0)
* Weighted loss to handle class imbalance
* Basic logging + static plots

### 🚧 Limitations of Initial System

* No real-time monitoring of training
* No visibility into **client behavior or drift**
* Hard to interpret **why performance drops under non-IID**
* No clinical usability (pure research code)
* No reproducible or replayable experiments

---

## 🚀 Phase 2 — Full Research System + Clinical Tool (Current Work)

We transformed the project into a **complete federated research + deployment platform**.

### Major Additions:

### 1️⃣ 📊 Real-Time Federated Dashboard

* Live training visualization
* Client-server network graph
* Accuracy vs rounds (live)
* Drift visualization per client
* Configurable experiments (α, clients, LR, algorithm)

👉 Now we **observe training dynamics**, not just final results.

---

### 2️⃣ 🔁 Replayable Experiment Engine

* Automatic JSON logging per round
* Replay past experiments like a simulation
* Enables reproducibility and debugging

👉 Converts experiments into **reusable research artifacts**

---

### 3️⃣ 📉 Drift-Aware Analysis

We introduced a key metric:

[
D_k = ||w_k - w_g||
]

* Measures how much each client deviates
* Directly linked to non-IID behavior
* Visualized in UI (edge thickness, charts)

👉 Turns federated learning into an **observable system**

---

### 4️⃣ 🧠 DriftAwareAgg

New aggregation strategy:

[
\alpha_k \propto \frac{n_k}{D_k}
]

* High-drift clients → lower influence
* Low-drift clients → higher weight

✔ Reduces instability
✔ Improves accuracy under non-IID
✔ Works better than FedAvg in skewed settings

👉 Key improvement over existing methods

---

### 5️⃣ 🏥 Clinical Diagnostic Interface

We extended beyond research:

* Upload fundus image
* DR grading (0–4)
* Confidence distribution
* Clinical findings + urgency
* Image quality warnings

👉 Makes system usable in **hospital/kiosk setting**

---

### 6️⃣ 🧪 Research Analysis Module

Dedicated analysis UI with:

* Algorithm comparison (FedAvg vs FedProx vs DriftAware)
* Non-IID degradation curves
* Convergence + drift behavior
* QWK (clinical metric)

👉 Enables **paper-level analysis directly from UI**

---

### 7️⃣ 🧬 Domain-Invariant Preprocessing

Pipeline added:

* Circular crop
* Ben Graham normalization
* CLAHE
* Standardization

👉 Removes dataset bias across hospitals
👉 Improves generalization

---

## 📊 Key Findings

* Performance degrades as non-IID increases (α ↓)
* FedProx stabilizes but doesn’t fully solve drift
* Drift correlates strongly with accuracy loss
* Drift-aware aggregation improves robustness

---

## 🧩 System Overview

```
Federated Training Engine
        ↓
JSON Logging + Checkpoints
        ↓
Live Dashboard (UI)
        ↓
Research Analysis
        ↓
Clinical Diagnostic Tool
```

This makes FRIED-RICE not just a model — but a **complete ecosystem**.

---

## 🏗️ What Makes This Different

Compared to a typical FL project:

| Feature                 | Typical | FRIED-RICE |
| ----------------------- | ------- | ---------- |
| Training                | ✔       | ✔          |
| Non-IID simulation      | ✔       | ✔          |
| Real-time visualization | ❌       | ✔          |
| Drift measurement       | ❌       | ✔          |
| Novel aggregation       | ❌       | ✔          |
| Replay system           | ❌       | ✔          |
| Clinical interface      | ❌       | ✔          |

---

## ▶️ How to Run

```bash
python -m ui.launch_ui
```

---

## 📌 Final Takeaway

This project evolved from:

> **“Train a federated model”**

into:

> **“Understand, analyze, and deploy federated learning in real-world medical systems”**

---

## 🚀 Future Scope

* Scale to 10–50 clients (real hospital simulation)
* Real GPU-based training integration
* Differential privacy
* Personalized federated learning
* Deployment as hospital monitoring system

---

