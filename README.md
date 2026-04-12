---
title: Perfect Venture
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Vexoo Labs AI Engineer Assignment

This repository contains the complete implementation of the AI Engineer Assignment for Vexoo Labs. It includes a 4-Layer Knowledge Pyramid for document ingestion, a semantic retrieval system, a fine-tuned reasoning model via LoRA, and an interactive web interface.

> **🔗 Live Interactive Demo:** [Hugging Face Spaces Demo](https://huggingface.co/spaces/Pranav20035/perfectventure)

---

## 📂 Project Structure

- `app.py`: A full Gradio web application providing a UI for the Knowledge Pyramid and Semantic Retrieval.
- `document_pyramid.py`: The architecture and data structures defining the 4-layer Knowledge Pyramid.
- `ingestion.py`: Implementation of the text chunking, sliding window, and document parsing mechanisms.
- `retrieval.py`: The semantic similarity search module powered by `sentence-transformers`.
- `train_gsm8k.py`: Fine-tuning pipeline using `trl` (version 1.0.0+) and `peft` (LoRA) against the GSM8K reasoning dataset.
- `bonus_adapter.md`: The conceptual system design for the Reasoning-Aware Adapter.
- `summary_report.md`: The 1-page summary of the technical choices and assignment breakdown.
- `requirements.txt`: Comprehensive list of Python package dependencies.

---

## 🛠️ Setup & Installation

The code has been strictly updated to ensure compatibility with modern library versions (e.g., `trl >= 1.0.0` requiring `SFTConfig`).

**1. Create a virtual environment (Optional but Recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**2. Install all required dependencies**
```bash
pip install -r requirements.txt
```
> **Note for Windows Users:** The `requirements.txt` will normally install standard PyTorch. Depending on your system's CUDA setup, you may need to install the CUDA-specific version of `torch` directly from the [PyTorch website](https://pytorch.org/).

---

## 🚀 Execution Guide

### 1. Launch the Live Web App (Part 1 - Recommended)
To experience the Document Ingestion and Retrieval system interactively, simply run the Gradio app:
```bash
python app.py
```
This will launch a local web server (usually at `http://127.0.0.1:7860`) where you can paste context and query the Knowledge Pyramid visually.

### 2. Run Scripts Manually via CLI (Part 1)
To view the raw CLI ingestion process:
```bash
python ingestion.py
```
To run the semantic retrieval search purely via terminal:
```bash
python retrieval.py
```

### 3. Fine-Tune the Reasoning Model (Part 2)
The GSM8K fine-tuning script is pre-configured and fully debugged. It gracefully handles CPU-only environments (falling back from `bf16/gpu`) and implements the correct `SFTConfig` parameters.
```bash
python train_gsm8k.py
```
*By default, this script uses `JackFram/llama-68m` to prevent out-of-memory errors on standard machines while demonstrating the mechanics.*

**To train against a full model like `meta-llama/Llama-3.2-1B`:**
1. Login via terminal: `huggingface-cli login`
2. Change `MODEL_NAME` inside `train_gsm8k.py` to your desired model.

---

## 📜 Bonus and Technical Summary

- **Summary Report:** See `summary_report.md` for the technical breakdown.
- **Bonus Question:** See `bonus_adapter.md` for the architectural design of a Reasoning-Aware Adapter.
