# DynaRoute: Cost-Efficient Adaptive Neural Routing for LLM Guardrails

<div align="center">

[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-StevenMup2004%2FDynaGuard--Data-ffd21e?style=for-the-badge)](https://huggingface.co/datasets/StevenMup2004/DynaGuard-Data)
[![Model on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-StevenMup2004%2FDynaRoute-ffd21e?style=for-the-badge)](https://huggingface.co/StevenMup2004/DynaRoute)
![Safety](https://img.shields.io/badge/Guardrail-DynaGuard%201.7B%20%2F%208B-blue?style=for-the-badge)
![Cost Saving](https://img.shields.io/badge/Inference%20Cost%20Saving-60.1%25-brightgreen?style=for-the-badge)

*An Agentic Adaptive Model Selection Framework inspired by SafeRoute, dynamically routing prompts between lightweight safety classifiers and high-capacity LLM guardrails.*

</div>

---

## 🤗 Hugging Face Hub Resources

All official dataset benchmarks, routing groundtruth targets, and trained Multi-Layer Perceptron (MLP) neural router checkpoints are publicly hosted on Hugging Face Hub:

* **📊 Evaluation & Training Dataset:** [datasets/StevenMup2004/DynaGuard-Data](https://huggingface.co/datasets/StevenMup2004/DynaGuard-Data)
  * Contains `train_router.json` (~3k+ training instances) and `test_router.json` (1,086 bilingual English/Vietnamese benchmark samples).
* **🧠 Neural Router Model Checkpoints:** [StevenMup2004/DynaRoute](https://huggingface.co/StevenMup2004/DynaRoute)
  * Contains trained PyTorch model weights (`model.pt`) optimized via Focal Loss for causal adaptive model escalation.

---

## 🌟 Executive Summary

As Large Language Models (LLMs) are deployed in agentic workflows, enforcing robust guardrails against jailbreaks, toxicity, and policy violations is paramount. However, production guardrails face a severe **Performance vs. Cost trade-off**:

1. **High-Capacity Models (`DynaGuard-8B`):** Achieve exceptional safety recognition accuracy but incur prohibitive inference costs and high latency.
2. **Lightweight Models (`DynaGuard-1.7B`):** Provide near-instantaneous inference at a fraction of the cost but are vulnerable to sophisticated contextual bypasses.

**DynaRoute** resolves this dilemma. Inspired by the *SafeRoute* methodology, DynaRoute introduces an intelligent neural router that intercepts hidden feature embeddings ($h \in \mathbb{R}^{2048}$) from the lightweight model. By accurately identifying "hard/unsafe" edge cases and forwarding *only* those queries to the 8B model, DynaRoute matches—and on multilingual datasets **outperforms**—the standalone 8B guardrail while cutting computational inference cost by **~60%**.

---

## 🧠 Methodology & Architecture

### Causal Pipeline Overview

```text
Input Prompt + Dynamic Policy
             |
             v
    Lightweight Guard (DynaGuard-1.7B)
             |
             +--- [Extract Hidden Features: h ∈ R^2048]
             |
             v
    Neural Router MLP (2048 -> 1024 -> 512 -> 256 -> 1)
          /     \
   P <= 0.60     P > 0.60 (Decision Threshold)
      /             \
   [Easy / Safe]   [Hard / Unsafe]
    |                  |
    v                  v
Fast Verdict     Heavy Guard (DynaGuard-8B)
(Cost = 1.7B)          |
                       v
                 Final Verdict
                 (Cost = 8B)
```

### 1. Oracle Groundtruth Labeling
To train the neural router without biased heuristics, we define an ideal **Oracle Groundtruth Assignment ($y \in \{0, 1\}$)**:
* **$y = 0$ (Easy Sample):** If `DynaGuard-1.7B` correctly predicts the safety groundtruth label ($Pred_{small} = GT$). Calling the small model is sufficient; zero additional inference cost is incurred.
* **$y = 1$ (Hard Sample):** If `DynaGuard-1.7B` fails ($Pred_{small} \neq GT$), but `DynaGuard-8B` succeeds ($Pred_{large} = GT$). The router is forced to escalate the query to the large model.

### 2. Router MLP & Loss Formulation
* **Architecture:** A 5-layer deep Multi-Layer Perceptron (`2048 -> 1024 -> 512 -> 256 -> 1`) with `BatchNorm1d`, `GELU` activations, and regularized `Dropout(0.3)`.
* **Focal Loss:** Severe class imbalance (easy samples vastly outnumber hard samples) is handled via Focal Loss ($\alpha=0.75, \gamma=2.0$), forcing the gradient optimization to focus heavily on misclassified hard instances.
* **Decision Threshold:** Fixed globally at **`FIXED_THRESHOLD = 0.60`** to maintain deterministic evaluation across all splits.

---

## 📂 Repository Structure

```text
Guardrail/fix-ver/
├── safe-route/
│   ├── models.py                   # BNN / MLP Router neural network definitions
│   ├── eval_end_to_end_safety.py   # E2E causal evaluation pipeline (Calculates F1 & Breakdown)
│   ├── eval_all_splits.py          # Validation / Test split evaluation script
│   ├── valid_router.json           # Router metadata and split tags
│   └── Plan.MD                     # Detailed executive presentation slide deck plan
├── DynaGuard/
│   ├── eval.py                     # Downstream safety test harness
│   └── ...                         # Feature extraction logs & hidden state buffers
└── .gitignore                      # Production gitignore excluding large weights & JSONs
```

---

## 🚀 Quickstart & Reproduction

All evaluation scripts have been upgraded with dynamic base directory resolution (`BASE_DIR`), enabling seamless execution from any working directory.

### Running End-to-End Safety Evaluation
```bash
# Execute pipeline verification across Original, Augmented, Test, and Valid splits
python safe-route/eval_end_to_end_safety.py
```

### Running Routing Threshold Diagnostics
```bash
# Evaluate router classification precision, recall, and AUC
python safe-route/eval_all_splits.py
```

---

## 💡 Future Roadmap
1. **Lightweight Pre-Routing Classifier:** Integrate a fast text-based Domain Classifier ($R_{domain}$). Incoming queries classified as *VN-native* are routed directly to `1.7B`; international or translated prompts trigger the `DynaRoute MLP`.
2. **Dynamic Threshold Tuning:** Enable runtime sliding thresholds based on server load and SLA budget constraints.

---
<div align="center">
<i>Built for Advanced Agentic Coding & Safety Guardrail Research.</i>
</div>
