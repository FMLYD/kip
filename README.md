# Kernel Point Imputation (KPI)

This repository contains the official implementation of **KPI: A Generalized Iterative Imputation Framework for Model Adaptation and Oracle Feature Utilization**.

## 🔍 Overview

Iterative imputation is widely used to handle missing data by sequentially imputing each feature using the others as predictors. However, existing methods suffer from:

1. **Model Misspecification**: Applying the same model structure to all features ignores the heterogeneity in their underlying data-generating processes.
2. **Underutilization of Oracle Features**: Fully observed (oracle) features are treated the same as partially missing ones, leading to suboptimal imputations.

To address these issues, we propose **Kernel Point Imputation (KPI)**, a **bi-level optimization framework** that:

- **Adapts model forms per feature** using functional optimization in a Reproducing Kernel Hilbert Space (RKHS).
- **Leverages oracle (fully observed) features** to provide informative supervision signals during the outer optimization stage.

## 🧠 Key Contributions

- A **flexible model adaptation** mechanism that avoids uniform parametric assumptions.
- A **bi-level learning framework** that exploits oracle features for more accurate and robust imputations.
- **Superior performance** across various real-world benchmarks with heterogeneous and partially observed datasets.

## 🛠️ Installation

```bash
cd KPI-imputation
```
You can run exper.py
