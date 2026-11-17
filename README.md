# JUST-AI-Academic-Assistant
# JUST-AI-Academic-Assistant

## Project Overview
This repository contains the Phase 2 implementation of a domain-specific GPT-2 model trained on the JUST BSc Data Science study plan. The project includes Continual Pre-training (CPT) and Supervised Fine-Tuning (SFT), followed by a comprehensive evaluation pipeline.

## Team Contributions
- **Rahaf Bani Salameh:** Corpus preparation, GPT-2 baseline.
- **Raghad Bani Salameh:** CPT training and evaluation.
- **Eman Obeidat:** SFT data formatting, SFT training & evaluation (`sft_results.csv`).
- **Ghaith Alhmoud:** Automated evaluation pipeline (Top-1/Top-5 accuracy, latency benchmarking), generation of SFT evaluation outputs, and Human Evaluation setup (scoring files generation).

## Evaluation and Results (Ghaith Alhmoud)
*(Note: The automated evaluation code was integrated into the main Kaggle Notebook. The results and scoring files are provided below).*

- **Automated Token-Level Accuracy:**
  - Base GPT-2: Top-1 = 47.01%, Top-5 = 64.94%
  - CPT Model: Top-1 = 55.78%, Top-5 = 75.30%
- **Latency Benchmarking (Inference Speed):**
  - Base GPT-2: 1.401s
  - CPT Model: 0.921s
  - SFT Model: 1.416s
- **Human Evaluation Setup:** Generated standardized scoring files (`base_scoring.csv`, `cpt_scoring.csv`, `sft_scoring.csv`) containing the 30 test questions with dedicated columns for manual rating.

## Uploaded Artifacts
- `accuracy_results.json`: Raw automated accuracy metrics.
- `base_vs_cpt_comparison.png`: Bar chart comparing Base and CPT models.
- `evaluation_pipeline.py`: Contains the specific code for automated evaluation and scoring generation.
