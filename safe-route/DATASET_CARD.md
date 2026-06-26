---
language:
- vi
- en
license: mit
task_categories:
- text-classification
- tabular-classification
tags:
- safety
- guardrails
- neural-routing
- agentic-ai
- cost-optimization
pretty_name: DynaRoute Adaptive Guardrail Routing Dataset
size_categories:
- 1K<n<10K
---

# DynaRoute Dataset: Adaptive Guardrail Routing Benchmark (Vietnamese & English)

This dataset contains the official training (`train_router.json`) and evaluation test (`test_router.json`) records designed to train and benchmark **DynaRoute Neural Routers**—an intelligent adaptive model selection mechanism inspired by *SafeRoute*.

---

## 📌 Dataset Overview

In agentic Large Language Model (LLM) serving pipelines, enforcing safety guardrails (detecting jailbreaks, toxicity, violent content, and policy violations) involves a strict **Cost vs. Accuracy trade-off**:
* **`DynaGuard-1.7B` (Small Model):** Ultra-fast, low inference cost, but vulnerable to translated/paraphrased edge cases.
* **`DynaGuard-8B` (Large Model):** High safety verification accuracy, but ~4x-5x more expensive computationally.

This dataset bridges the gap by providing deterministic **Oracle Groundtruth Routing Targets**, constructed by empirical evaluation across English DynaBench and Vietnamese translated benchmarks (`montehoover/DynaBench` Original and `dynabench_latest` Augmented).

---

## 🗂 Files & Splits

* **`train_router.json`**: Training split (~3,000+ samples) used to train the 5-layer Multi-Layer Perceptron (MLP) router via Focal Loss.
* **`test_router.json`**: **Test Benchmark Split (1,086 samples)** representing the core Vietnamese downstream test set (543 Original + 543 Augmented samples).

---

## 📋 Data Fields & Annotation Schema

Each JSON object in the dataset contains the following attributes:

* **`prompt`** *(string)*: The user prompt or incoming instruction being inspected.
* **`response`** *(string)*: The generated assistant response (if evaluating input-output safety).
* **`source`** *(string)*: Origin tag indicating the benchmark subset (`montehoover/DynaBench` or `dynabench_latest`).
* **`small_pred`** *(integer)*: Binary safety prediction of the lightweight model `DynaGuard-1.7B` (`0`: PASS/Safe, `1`: FAIL/Unsafe).
* **`large_pred`** *(integer)*: Binary safety prediction of the high-capacity model `DynaGuard-8B` (`0`: PASS, `1`: FAIL).
* **`ground_truth`** *(integer)*: The definitive human-annotated safety groundtruth label (`0`: PASS, `1`: FAIL).
* **`label`** *(integer)*: **Binary Oracle Routing Target ($y \in \{0, 1\}$)**.

---

## 🏷 Oracle Routing Target Definition (`label`)

The binary target **`label`** dictates the optimal routing decision to achieve maximum accuracy at minimum cost:

$$\text{label} = \begin{cases} 
0 & \text{if } Pred_{small} == GT \quad \text{(Easy: Small model is accurate; save 8B cost)} \\
1 & \text{if } Pred_{small} \neq GT \text{ and } Pred_{large} == GT \quad \text{(Hard: Escalation required)} \\
0 & \text{otherwise}
\end{cases}$$

* **`label = 0` (Locally Resolvable):** `DynaGuard-1.7B` gets the correct answer. The system returns the fast verdict immediately (saving 100% of the 8B computational cost).
* **`label = 1` (Must Escalate):** `DynaGuard-1.7B` gets fooled, but `DynaGuard-8B` knows the truth. The router must forward the prompt to the heavy guard.

---

## 🏆 Empirical Downstream Performance

When training an MLP Router (`2048 -> 1024 -> 512 -> 256 -> 1`) on this dataset and operating at threshold **`0.60`**:
* **Safety F1:** DynaRoute achieves **0.7542** on the Original Vietnamese split—**outperforming standalone DynaGuard-8B** (`0.7320`).
* **Cost Efficiency:** Reduces calls to the expensive 8B model by **60.1%**.

---
*Curated for Advanced Agentic Coding & Safety Guardrail Research.*
