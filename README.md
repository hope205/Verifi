# Verifi
A Parameter-Efficient Fine-Tuned model  for Financial Reasoning



# 🏦 Verifi: Precision Financial LLM via LoRA


[![Model: Llama-3](https://img.shields.io/badge/Model-Llama--3--8B-blue)](https://huggingface.co/meta-llama)
[![Framework: Hugging Face](https://img.shields.io/badge/Framework-Hugging%20Face-orange)](https://huggingface.co/docs/peft/index)
[![Dataset: FinQA](https://img.shields.io/badge/Dataset-FinQA-green)](https://github.com/google-research-datasets/finqa)

## 🎯 Project Objective
General-purpose LLMs often fail at financial questions hidden within dense reports. **Verifi** addresses this by fine-tuning the model to recognize financial structures and generate answers quantitative queries.

## 🛠️ Tech Stack
* **Base Model:** Meta-Llama-3-8B-Instruct
* **Fine-Tuning:** LoRA (Low-Rank Adaptation) via Hugging Face `PEFT`
* **Training:** `TRL` (SFTTrainer) with 4-bit Quantization (QLoRA)
* **Dataset:** FinQA (Financial Question Answering)
