# Kernel Point Imputation (KPI)

This repository contains the official implementation of **KPI: A Generalized Iterative Imputation Framework for Model Adaptation and Oracle Feature Utilization**.

## 🔍 Overview
![KPI Method Overview](./method.png)

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



## 🧪 Reproducing Results

### 1. Environment Setup
Install Python 3.10 and required dependencies:
```bash
pip install -r requirements.txt
```


### 2. Data Preparation
Obtain preprocessed datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu), then place the downloaded data in the `./dataset` folder.

Supported datasets summary:
<img width="1143" height="431" alt="image" src="https://github.com/user-attachments/assets/f416f80c-6ebc-445c-87a4-63948ccb410b" />


### 3. Train and Evaluate
Experiment scripts for all benchmarks are provided under the `./scripts/` folder. Reproduce results with example commands like:
```bash
bash scripts/bt.sh
```



## Acknowledgements

We gratefully acknowledge the UCI Machine Learning Repository for providing the datasets used in this work. The availability of high-quality, well-documented, and openly accessible datasets is critical to the development, evaluation, and reproducibility of machine learning research. We appreciate the efforts of the UCI community in curating and maintaining this valuable resource, which continues to support and accelerate progress in the broader data science community.





