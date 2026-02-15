# Federated Learning Analysis

## Experimental Setup

- 3 clients
- Extreme non-IID partitioning (label-sorted shards)
- FedAvg aggregation
- Local epochs: 1
- Rounds: 20
- Weighted loss in clients

---

## Results

### Centralized Accuracy
85.79%

### Federated Accuracy (Stabilized)
Peak: ~78%
Final: ~75–76%

Gap: ~7–8%

---

## Observations

1. Extreme non-IID causes client drift.
2. Naive FedAvg oscillates heavily.
3. Weighted aggregation improves stability.
4. Federated performance remains below centralized baseline.

---

## Theoretical Insight

Federated learning minimizes:

min Σ_k p_k F_k(w)

Under non-IID conditions, local objectives F_k differ significantly, leading to inconsistent gradient directions and slower convergence.

---

## Key Insight

Performance degradation is primarily driven by:

- Data heterogeneity
- Minority class underrepresentation
- Client drift during local updates

---
