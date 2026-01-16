
# Mitigating Hallucinations in VLMs using SPIN x MOD

## 1. Technical Specifications

### 1.1 Evaluation Framework (LLM Judge)

- **Judge Model:** Qwen/Qwen2.5-3B-Instruct (Local Inference)
- **Role:** Single-answer grading (0-6 scale) & Hallucination detection
- **Parameters:**
  - **temperature:** 0.0 (Deterministic)
  - **do_sample:** False
  - **max_new_tokens:** 256
- **Prompt Template:** Standardized "Impartial AI Judge" system prompt with 5-shot examples.

### 1.2 Vision-Language Model (VLM) Configuration

- **Base Architecture:** llava-hf/llava-1.5-7b-hf
- **LLM Backbone:** Vicuna-1.5-7b
- **Vision Tower:** CLIP-ViT-L-336px

#### Mitigation Method 1: SPIN (Image-Guided Head Suppression)

- **Type:** Inference-Time Attention Patch (Runtime Injection)
- **Mechanism:** Dynamic suppression of attention heads with low relevance to image tokens during the decoding phase.
- **Keep Head Ratio:** 0.95 (Retains top 95% of heads)
- **Suppression Alpha:** 0.08 (Suppressed heads are scaled by this factor)
- **Activation Condition:** Active only during decoding (**q_len == 1**).

#### Mitigation Method 2: SPIN + MOD (Dynamic Contrastive Decoding)

- **Mechanism:** Calculates Jensen-Shannon (JS) Divergence between vision-conditioned logits and text-only logits.
- **Logic:**
  - If **JS Divergence < Threshold (Grounded):** Uses standard Greedy Decoding.
  - If **JS Divergence >= Threshold (Hallucination Risk):** Uses Contrastive Decoding (**Vision Logits - Alpha * Text Logits**).
- **JS Threshold:** 0.08
- **Alpha Contrastive:** 0.5

### 1.3 Inference Settings

- **Decoding Strategy:** Greedy Decoding (Baseline/SPIN) / Dynamic Contrastive (MOD)
- **Temperature:** 0.0
- **Repetition Penalty:** 1.1
- **Max New Tokens:** 128
- **Prompt Format:** `USER: <image>\n{question}\nASSISTANT:`

---

## 2. Executive Summary

This document outlines the technical implementation and evaluation results of applying SPIN (Image-Guided Head Suppression) and Dynamic Contrastive Decoding (MOD) to the LLaVA-v1.5 Vision-Language Model. The objective was to assess whether suppressing specific attention heads or dynamically penalizing text-only priors could reduce hallucinations on the MMHal-Bench benchmark.

While the initial SPIN implementation resulted in a performance regression, adding the MOD (Dynamic Contrastive Decoding) layer significantly recovered performance in specific categories like Environment and Relation, though the overall score remains slightly below the strong baseline.

---

## 3. Implementation Overview

### 3.1 System Architecture

The solution is built upon the Hugging Face Transformers library, utilizing a custom forward hook to modify attention behavior and a custom decoding loop for dynamic logit manipulation.

- **Base Model:** llava-1.5-7b-hf loaded in float16.
- **SPIN Patch:** A custom `llama_spin_forward` function replaces the standard attention method. It calculates the sum of attention probabilities assigned to image tokens and suppresses heads that fall below the top 95% threshold.
- **MOD Decoding:** A custom generation loop (`dynamic_decode_one_sample`) that runs parallel forward passes (Vision+Text vs. Text-Only) and dynamically adjusts the next-token selection strategy based on divergence metrics.
- **Benchmark:** MMHal-Bench (96 samples), evaluating across 8 categories including Attribute, Adversarial, and Counting.

### 3.2 Evaluation Pipeline

The workflow defined in `llava1-5-x-spin on mmhalbench.ipynb` proceeds as follows:

1. **Baseline Inference:** Run LLaVA-1.5 with standard settings.
2. **SPIN Inference:** Inject the SPIN patch (`use_spin_img=True`).
3. **SPIN + MOD Inference:** Run the model with both the SPIN patch and the Dynamic Contrastive Decoding loop.
4. **Judgement:** Use Qwen2.5-3B-Instruct to act as an impartial judge, scoring responses on accuracy and hallucination presence.

---

## 4. Experimental Results

The following data compares the performance of the model across three configurations: Baseline, With SPIN, and With SPIN + MOD.

### 4.1 High-Level Metrics

| Metric             | With SPIN + MOD | With SPIN | Baseline (No SPIN) | Delta (MOD vs Base) |
| ------------------ | --------------: | --------: | -----------------: | ------------------: |
| Total Samples      |              96 |        96 |                 96 |                   - |
| Average Score      |            2.70 |      2.45 |               2.86 |               -0.16 |
| Hallucination Rate |            0.46 |      0.49 |               0.43 |               +0.03 |

**Note:** Higher Average Score is better; Lower Hallucination Rate is better.

### 4.2 Category-Specific Performance

The benchmark evaluates 8 distinct categories. The table below details the average score (0-6 scale) for each.

| Category    | With SPIN + MOD | With SPIN | Baseline | Analysis                      |
| ----------- | --------------: | --------: | -------: | ----------------------------- |
| Holistic    |            4.17 |      4.08 |     3.83 | ✅ Best Performance (+0.34)   |
| Environment |            4.67 |      2.67 |     4.08 | ✅ Best Performance (+0.59)   |
| Relation    |            3.00 |      2.17 |     2.17 | ✅ Best Performance (+0.83)   |
| Other       |            2.75 |      2.08 |     2.67 | ✅ Slight Improvement (+0.08) |
| Comparison  |            1.83 |      2.92 |     2.92 | ❌ Significant Drop           |
| Counting    |            1.83 |      2.08 |     2.67 | ❌ Degraded                   |
| Attribute   |            2.00 |      2.08 |     3.17 | ❌ Degraded                   |
| Adversarial |            1.33 |      1.50 |     1.42 | ➖ Neutral                    |

---

## 5. Analysis & Observations

### 5.1 Recovery with MOD

The addition of Dynamic Contrastive Decoding (MOD) partially recovered the performance loss observed with SPIN alone.

- **Environment & Relation:** The MOD strategy resulted in massive improvements in Environment (4.67 vs 2.67) and Relation (3.00 vs 2.17), outperforming even the baseline. This suggests the contrastive penalty effectively reduces hallucinations in spatial and contextual reasoning.
- **Holistic:** The model achieved its highest score (4.17) in this category, indicating excellent summarization capabilities.

### 5.2 Persistent Weaknesses

Despite the improvements, the combined SPIN + MOD approach suffered in specific granular tasks:

- **Counting & Comparison:** Scores dropped significantly (Counting: 1.83, Comparison: 1.83). The contrastive penalty might be over-penalizing valid tokens in scenarios requiring strict numerical or comparative precision.
- **Attribute:** Performance remains low (2.00 vs Baseline 3.17), indicating difficulty in correctly binding attributes to objects.

### 5.3 Conclusion

The SPIN + MOD configuration creates a "specialist" profile. While it lags slightly behind the baseline in overall general score (2.70 vs 2.86), it demonstrates superior grounding in complex Environment and Relation tasks. Future work should focus on tuning the `alpha_contrastive` parameter or `js_threshold` specifically for counting and attribute-based queries to achieve a universally superior model.
