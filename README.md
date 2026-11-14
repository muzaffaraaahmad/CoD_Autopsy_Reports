# Cause-of-Death Prediction from Verbal Autopsies with QLoRA-Fine-Tuned LLMs

## Overview

This project fine-tunes six open-source LLMs to predict **cause of death** from verbal autopsy records in the **PHMRC Gold Standard** dataset. Each model is trained with **QLoRA** (4-bit quantized low-rank adaptation), with a separate adapter for each age module (adult, child, neonate).

**Models:** Qwen2.5-7B-Instruct, Qwen3-14B, Phi-4, DeepSeek-R1-Distill-Qwen-14B, BioMistral-7B, Gemma-3-12B-it

## How it works

1. **Preprocess:** each structured yes/no questionnaire record is turned into a short text narrative (sex, age, up to 30 affirmed symptoms, up to 25 keywords).
2. **Split:** records are split 70/15/15 into train, validation and test sets, stratified by cause.
3. **Fine-tune:** QLoRA (4-bit NF4, LoRA rank 32) with loss computed only on the cause-of-death text.
4. **Predict:** every candidate cause is scored by log-likelihood and ranked.
5. **Evaluate:** Top-1, Top-5 and CSMF accuracy, plus F1 and ROC AUC.

## Quick start

1. Download the PHMRC adult, child and neonate CSVs and the codebook from [IHME GHDx](https://ghdx.healthdata.org/) (data not included).
2. Set `DATA_PATHS`, `CODEBOOK_PATH` and `MODEL_NAME` in the notebook's Configuration cell.
3. Run all cells on a CUDA GPU.

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate scikit-learn pandas matplotlib seaborn openpyxl scipy
```

## Results

| Model | Module | Top-1 | Top-5 | CSMF |
|---|---|---|---|---|
| | | | | |

## Note

Research prototype only. Not for clinical use or official mortality coding without independent validation.
