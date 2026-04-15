# Scaling Laws for Neural Language Models
**Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020)**  
*OpenAI — arXiv:2001.08361*

---

## Overview

This paper investigates how the performance of Transformer-based language models depends on three key factors:

- **N** — Number of model parameters (excluding embeddings)
- **D** — Dataset size in tokens
- **C** — Compute budget used during training (measured in PF-days)

The central finding: **performance follows smooth power-law relationships with each of these factors, spanning over seven orders of magnitude.** Architectural details like depth vs. width matter very little — scale is what drives improvement.

The most influential practical conclusion: **when given a fixed compute budget, you should train a much larger model and stop early, rather than training a smaller model to convergence.**

This paper directly informed the scaling strategy behind GPT-3 and subsequent large language models.

---

## Architecture Overview (Pseudocode)

This paper does not introduce a new model architecture. Instead, it introduces a **scaling law framework** — a predictive system for language model performance. Below is a formal pseudocode representation of the framework.

```
# ============================================================
# SCALING LAWS FRAMEWORK — Kaplan et al. (2020)
# ============================================================

# --- INPUTS ---
N  ← number of non-embedding model parameters
D  ← dataset size in tokens
C  ← total training compute (PF-days), where C ≈ 6 * N * B * S
     (B = batch size, S = training steps)

# --- CORE POWER LAW EQUATIONS ---

# Loss as a function of model size (trained to convergence, infinite data)
L(N) = (Nc / N)^αN
    where αN ≈ 0.076,  Nc ≈ 8.8e13 parameters

# Loss as a function of dataset size (large model, early stopping)
L(D) = (Dc / D)^αD
    where αD ≈ 0.095,  Dc ≈ 5.4e13 tokens

# Loss as a function of compute (optimally allocated)
L(Cmin) = (Cmin_c / Cmin)^αCmin
    where αCmin ≈ 0.050,  Cmin_c ≈ 3.1e8 PF-days

# --- JOINT SCALING LAW (N and D together) ---
L(N, D) = [ (Nc/N)^(αN/αD) + (Dc/D) ]^αD
    # When D → ∞, this reduces to L(N)
    # When N → ∞, this reduces to L(D)

# --- TRAINING CURVE SCALING (N and training steps S) ---
L(N, S) = (Nc/N)^αN + (Sc/Smin)^αS
    where αS ≈ 0.76,  Sc ≈ 2.1e3 steps

# --- OPTIMAL COMPUTE ALLOCATION ---
# Given a fixed budget Cmin, solve for optimal N, B, S:

FUNCTION optimal_allocation(Cmin):
    N_opt  ∝ Cmin^0.73   # Model size grows fastest
    B_opt  ∝ Cmin^0.24   # Batch size grows moderately  
    S_opt  ∝ Cmin^0.03   # Training steps barely change

    RETURN N_opt, B_opt, S_opt

# KEY INSIGHT: As compute increases, spend it on BIGGER MODELS
# not more training steps. Stop training significantly before convergence.

# --- CRITICAL BATCH SIZE ---
Bcrit(L) = B* / L^(1/αB)
    where B* ≈ 2e8 tokens,  αB ≈ 0.21
    # Bcrit depends only on loss, not model size

# --- OVERFITTING CONDITION ---
# To avoid overfitting when scaling model size:
D ≳ (5e3) * N^0.74
    # Dataset only needs to grow sublinearly with model size

# --- EXPERIMENTAL METHODOLOGY ---
FOR each scale factor X in {N, D, C}:
    fix the other two factors
    train models across 6-8 orders of magnitude of X
    measure cross-entropy loss L on WebText2 test set
    fit power-law: L(X) = (Xc / X)^αX
    verify trend holds across all scales

# --- ARCHITECTURE TESTED ---
Model: Decoder-only Transformer (GPT-style)
       [VSP+17, same as GPT-2 architecture]
Optimizer: Adam (small models), Adafactor (>1B params)
Context: 1024 tokens
Dataset: WebText2 (22B tokens, web text filtered by Reddit karma)
Models: 768 params → 1.5 billion params
```

### How This Differs from Prior Work

| Aspect | Prior Assumption | Kaplan et al. Finding |
|---|---|---|
| Architecture shape | Depth/width matter significantly | Minimal effect when N is fixed |
| Training strategy | Train small models to convergence | Train large models, stop early |
| Data scaling | Linear with model size | Sublinear: D ∝ N^0.74 |
| Performance prediction | Hard to predict | Predictable power laws |
| Optimal compute use | Unclear | Spend on model size, not steps |

---

## Critical Analysis

### What the paper does well
- Massive empirical scope — hundreds of training runs across 8 orders of magnitude
- Power-law fits are remarkably clean and consistent
- Practical, actionable conclusions for compute allocation

### Limitations and open questions

**1. No theoretical foundation**  
The authors explicitly admit they have no deep theory for *why* these power laws exist. They describe it as analogous to empirical gas laws before thermodynamics existed. Without theory, we don't know when or why the laws might break down.

**2. Single architecture, single dataset**  
All experiments use decoder-only Transformers on WebText2. It's unclear whether the exact exponents transfer to other architectures (BERT-style encoders, diffusion models, MoE models) or other data domains (code, images, clinical text).

**3. The Chinchilla correction (2022)**  
Hoffmann et al. (DeepMind, 2022) reran scaling law experiments with better-controlled methodology and found that Kaplan et al. *underestimated* the importance of data*. The original paper suggested training very large models on relatively little data. Chinchilla showed that **model size and data should scale equally** — a significant correction that changed how GPT-4 and subsequent models were trained.

**4. Environmental costs not addressed**  
The paper advocates for ever-larger models but does not address the environmental cost of training at scale. A single large training run can emit CO2 equivalent to multiple transatlantic flights. As scaling laws encourage bigger models, this becomes a serious omission.

**5. Emergent capabilities not predicted**  
The smooth power-law framing misses a phenomenon discovered later: at certain scale thresholds, models suddenly gain qualitatively new capabilities (Wei et al., 2022). Power laws suggest smooth, predictable improvement — but emergence is discontinuous and was not anticipated here.

**6. Fixed tokenization and vocabulary**  
The authors note that constants like Nc and Dc depend on vocabulary size and tokenization and "have no fundamental meaning." This limits the generalizability of the precise numerical values.

---

## Impact

### Immediate impact (2020)
- **Directly motivated GPT-3** (Brown et al., 2020) — 175B parameters, trained the same year
- Gave AI labs a **quantitative roadmap** for compute allocation
- Shifted the field from "train to convergence" to "train large, stop early"
- Established that **architecture tweaks matter far less than scale**

### Medium-term impact
- Inspired **Chinchilla (2022)** which corrected the data/model balance
- Motivated research into **compute-optimal training**
- Influenced investment decisions — if performance is predictable, ROI on compute is predictable
- Sparked debate about whether scale is the only path forward

### Present and future
- Scaling laws have been extended to **multimodal models**, **code**, and **reasoning tasks**
- The paper's framework is now standard vocabulary in ML research
- **Efficiency research** (LoRA, quantization, MoE, Mamba) is partly a response to the environmental and financial costs of the scaling-first strategy this paper advocates
- The question of whether scaling laws hold indefinitely — or whether we need architectural innovation — is one of the most active debates in AI today

---

## Two Discussion Questions

**Question 1:**  
The paper shows that for a fixed compute budget, you should train a much larger model and stop early rather than training a smaller model to convergence. Why do you think most practitioners *before* this paper were doing the opposite — and what would have to change in your own workflow or intuitions to adopt the compute-efficient approach?

**Question 2:**  
Chinchilla (2022) found that Kaplan et al. underweighted the importance of training data — suggesting models like GPT-3 were undertrained relative to their size. Given that both papers use empirical power-law fitting on Transformers, what does it tell us about relying on empirical scaling laws without theoretical foundations?

---

## Resource Links

1. **Original Paper** — https://arxiv.org/abs/2001.08361
2. **Chinchilla Scaling Laws (DeepMind, 2022)** — https://arxiv.org/abs/2203.15556 *(the key correction to this paper)*
3. **GPT-3 Paper (direct application of these laws)** — https://arxiv.org/abs/2005.14165
4. **Emergent Abilities of Large Language Models (Wei et al., 2022)** — https://arxiv.org/abs/2206.07682 *(what scaling laws missed)*
5. **The FLOPs Calculator — Understanding Compute** — https://github.com/google-research/google-research/tree/master/scaling_transformer_inference_efficiency

---

## Code Demonstration

See `scaling_demo.py` — a Python script that empirically demonstrates the power-law relationship between model size and loss, and visualizes the compute-efficient frontier.

---

## Citation

```
@article{kaplan2020scaling,
  title={Scaling Laws for Neural Language Models},
  author={Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B and Chess, Benjamin and Child, Rewon and Gray, Scott and Radford, Alec and Wu, Jeffrey and Amodei, Dario},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}
```

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361.
