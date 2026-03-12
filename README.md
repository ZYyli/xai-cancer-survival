# XAI-Cancer-Survival
A Systematic Evaluation Framework for Explainable AI (XAI) in Deep Learning-based Cancer Survival Prediction

📖 Introduction
This project provides a systematic evaluation framework to assess the performance of various Explainable AI (XAI) methods within the context of cancer survival prediction. Utilizing transcriptome data from 15 major cancer types in The Cancer Genome Atlas (TCGA), we constructed Self-Normalizing Neural Network (SNN) models and performed a comparative analysis across six core XAI algorithms.

🧪 Core Features
15 Cancer Types: Including BRCA, LUAD, LUSC, KIRC, and more.

6 XAI Methods:
Integrated Gradients (IG)
GradientSHAP
DeepSHAP (A unified approach combining DeepLIFT and SHAP)
DeepLIFT
Layer-wise Relevance Propagation (LRP)
Permutation Feature Importance (PFI)
Evaluation Metrics: Biological consistency, prognostic factor enrichment analysis, and stability assessment.

📁 Project Structure
datasets_csv/: Data preprocessing scripts (normalization, sample splitting, etc.).
operation/: Core XAI evaluation algorithms and Bootstrap validation scripts.
biological_plausibility/: Scripts for biological functional validation and visualization logic.

🚀 Quick Start
1. Environment Setup
Ensure you have Python installed, then install the required dependencies:
Bash
<pip install -r requirements.txt>
2. Data Preprocessing
Run the following script to prepare the TCGA transcriptome data:

Bash
python datasets_csv/preprocessing_cancer_single.py
3. Running XAI Evaluation
To perform the feature stability analysis using Bootstrap, execute:

Bash
python operation/feature_stability_analysis_bootstrap.py
