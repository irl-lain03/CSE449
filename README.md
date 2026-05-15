# A Parallel Machine Learning Pipeline for Text Sentiment Classification

> **Course:** CSE449 — Parallel and High-Performance Computing  
> **Institution:** BRAC University, Dhaka, Bangladesh  
> **Author:** Fatima Rahman 
> **Session:** Spring 2026  

---

## 📌 Executive Summary
In Natural Language Processing (NLP), classical machine learning pipelines (e.g., Naive Bayes, Support Vector Machines, Random Forests) remain competitive, highly interpretable, and cost-effective alternatives to deep learning architectures without requiring specialized GPU hardware. However, typical implementations execute sequentially on a single thread—leaving modern multi-core processors largely idle (often utilizing only ~8% of total CPU capacity on a 12-thread machine).

This repository contains the design, implementation, and empirical evaluation of a **fully parallelized machine learning pipeline** for binary text sentiment classification. Parallelism is applied across three distinct stages:
1. **Parallel Preprocessing & Tokenization** (`joblib`)
2. **Parallel Base Model Training** (`joblib` with concurrent dispatch)
3. **Parallel Hyperparameter Optimization (HPO)** (`Optuna`)
4. **Stacking Ensemble Meta-Learner & Explainability** (Logistic Regression + `SHAP`)

The pipeline was benchmarked across the complete spectrum of **1 to 12 CPU workers** with **5 independent repetitions** per worker configuration to rigorously evaluate speedup, parallel efficiency, and coordination overhead.

---

## 🔬 Benchmark & Empirical Results

All experiments were conducted on an **AMD Ryzen 5 7600 CPU** (6 physical cores, 12 logical processors via SMT, 32 GB RAM) on Windows 11 with Python 3.10.11, incorporating deliberate cooling periods between runs to eliminate thermal throttling artifacts.

### 1. Scaling & Speedup Overview
| Pipeline Stage | 1 Worker (Sequential Baseline) | 12 Workers (Max Parallel) | Peak Speedup | Coordination Overhead |
|:---|:---:|:---:|:---:|:---:|
| **Data Preprocessing** | 4.217s (±0.028s) | 0.765s (±0.053s) | **5.51×** | 0.14% → 0.84% |
| **Model Training** | 101.053s (±2.172s) | 13.877s (±0.283s) | **7.28×** | Efficiency: 60.7% |
| **Optuna HPO** | 23.892s (±5.835s) | 11.475s (±4.752s) | **2.08×** | Best F1: 0.9033 |

> **Hardware Insight:** A minor efficiency drop was observed at worker 11 (speedup 7.00× vs. 7.22× at worker 10). With 5 repetitions demonstrating a low standard deviation (±0.425s), this was identified as authentic OS scheduler behavior: mapping 11 concurrent threads across 12 logical processors creates asymmetric core allocation and thread migration stalls. At 12 workers, symmetric mapping across all logical processors restores optimal load balancing.

---

### 2. Model Performance Comparison

Evaluated on the **IMDb Movie Reviews dataset** (50,000 reviews, 80/20 train-test split):

| Model / Architecture | Precision | Recall | F1 Score | AUC-ROC |
|:---|:---:|:---:|:---:|:---:|
| Multinomial Naive Bayes | 0.88 | 0.89 | 0.8845 | — |
| Linear SVC | 0.91 | 0.91 | 0.9068 | — |
| Random Forest (100 Trees) | 0.86 | 0.86 | 0.8632 | — |
| **Stacking Ensemble (Meta-Learner)** | **0.91** | **0.91** | **0.9121** | **0.9118** |

* **Zero Accuracy Trade-off (Deterministic Execution):** A paired *t*-test between sequential ($n\_jobs=1$) and parallel ($n\_jobs=-1$) 5-fold cross-validation runs produced identical F1 score arrays (`[0.9016, 0.9039, 0.9008, 0.9001, 0.9030]`, absolute difference = `0.000000`). Parallelization introduced zero stochastic variance or degradation in accuracy.
* **Cross-Domain Generalization:** The pipeline was tested on an unseen external validation dataset (**50,000 Amazon Product Reviews**) with zero retraining. The stacking ensemble retained **F1 = 0.8370** and **AUC-ROC = 0.8262**, demonstrating strong portability across distinct review vocabularies.

---

## 📈 Benchmark & Evaluation Plots

### Scaling & Hardware Performance
| Preprocessing Speedup & Coordination Overhead | Model Training Speedup & Parallel Efficiency |
|:---:|:---:|
| ![Preprocessing Speedup](assets/preprocessing_speedup.png) | ![Training Speedup](assets/training_speedup.png) |

---

### Confusion Matrices
| Primary Evaluation: IMDb Test Set (F1: 0.9121) | External Validation: Amazon Reviews (F1: 0.8370) |
|:---:|:---:|
| ![IMDb Confusion Matrix](assets/confusion_matrix.png) | ![Amazon Confusion Matrix](assets/amazon_confusion_matrix.png) |

---

### Interpretability & Explainability (SHAP)
SHAP (SHapley Additive exPlanations) was applied to the Logistic Regression meta-learner across 300 test instances to identify top predictive drivers:
- **Positive Sentiment Drivers:** `great`, `best`, `love`, `well`, `excellent`
- **Negative Sentiment Drivers:** `bad`, `worst`, `nothing`, `awful`, `waste`

<p align="center">
  <img src="assets/shap_plot.png" alt="SHAP Feature Importance" width="750"/>
</p>

---

## 📂 Repository Structure

```text
parallel-sentiment-pipeline/
├── assets/
│   ├── eda_dist.png                   # Dataset class balance
│   ├── eda_length.png                 # Review length distribution
│   ├── eda_top_words.png              # Top-20 word frequencies
│   ├── preprocessing_speedup.png      # Preprocessing scaling curve
│   ├── training_speedup.png           # Training scaling & efficiency
│   ├── confusion_matrix.png           # Stacking ensemble matrix (IMDb)
│   ├── amazon_confusion_matrix.png    # External validation matrix (Amazon)
│   └── shap_plot.png                  # SHAP global feature impact
├── notebooks/
│   └── Code_cse449_22201726_Group09.ipynb
├── requirements.txt
├── .gitignore
└── README.md