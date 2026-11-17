# KnowledgeForge – Semantic Knowledge Retrieval & LLM Fine-Tuning

KnowledgeForge is an AI-powered application that combines hierarchical document organization, semantic retrieval, and parameter-efficient LLM fine-tuning for mathematical reasoning.

It provides an interactive interface for processing text, retrieving relevant information, and exploring a fine-tuning pipeline using LoRA.

## Key Features

- **Document Ingestion:** Processes text and organizes content into manageable sections.
- **Four-Layer Knowledge Pyramid:** Structures document information across four hierarchical layers.
- **Semantic Retrieval:** Uses sentence embeddings and similarity search to retrieve relevant content.
- **Text Chunking:** Implements chunking and sliding-window techniques for document processing.
- **LLM Fine-Tuning:** Uses LoRA to fine-tune a pretrained language model.
- **Mathematical Reasoning:** Uses the GSM8K dataset for model fine-tuning.
- **Interactive Interface:** Provides a Gradio application for exploring document ingestion and retrieval.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Interface | Gradio |
| Embeddings | Sentence Transformers |
| Retrieval | Semantic Similarity Search |
| Model Fine-Tuning | LoRA, PEFT |
| Training Framework | TRL |
| Dataset | GSM8K |
| Base Model | Llama |

## System Architecture

KnowledgeForge consists of two main components.

### 1. Document Knowledge and Retrieval

The document processing pipeline follows these stages:

1. **Ingestion:** Accepts text content for processing.
2. **Chunking:** Splits documents into smaller sections using sliding-window techniques.
3. **Knowledge Organization:** Structures information using a four-layer Knowledge Pyramid.
4. **Embedding Generation:** Converts text into vector representations using Sentence Transformers.
5. **Semantic Retrieval:** Finds relevant content based on semantic similarity.

### 2. LLM Fine-Tuning

The fine-tuning pipeline adapts a pretrained language model for mathematical reasoning.

1. Loads a pretrained Llama model.
2. Prepares the GSM8K mathematical reasoning dataset.
3. Applies LoRA adapters using PEFT.
4. Configures supervised fine-tuning using TRL.
5. Trains the model and saves the resulting adapter.

The document retrieval and model fine-tuning pipelines are maintained as separate components.

## Project Structure

```text
KnowledgeForge/
├── app.py
├── document_pyramid.py
├── ingestion.py
├── retrieval.py
├── train_gsm8k.py
├── bonus_adapter.md
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- Hugging Face account for accessing gated models, if required

### 1. Clone the Repository

```bash
git clone <repository-url>
cd KnowledgeForge
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
python app.py
```

The Gradio application will be available locally, usually at:

```text
http://127.0.0.1:7860
```

## Running Individual Modules

Run document ingestion:

```bash
python ingestion.py
```

Run semantic retrieval:

```bash
python retrieval.py
```

## Model Fine-Tuning

The GSM8K fine-tuning pipeline uses LoRA to reduce the number of trainable parameters during model adaptation.

Run the training script:

```bash
python train_gsm8k.py
```

The default configuration uses `JackFram/llama-68m` to keep resource requirements relatively low.

To experiment with a larger model, update `MODEL_NAME` in `train_gsm8k.py` and ensure your hardware has sufficient memory.

For gated models, authenticate with Hugging Face before running the training script.

The training configuration includes CPU-compatible fallbacks for environments where GPU-specific precision options such as BF16 are unavailable.

## Architecture Exploration

The repository also includes `bonus_adapter.md`, which documents a conceptual design for a Reasoning-Aware Adapter and its potential application to reasoning-oriented language models.

## Future Improvements

- Integrate the retrieval pipeline with a conversational LLM for complete RAG-based question answering.
- Improve hierarchical retrieval across the four knowledge layers.
- Evaluate retrieval quality using relevant information-retrieval metrics.
- Compare base and fine-tuned model performance on mathematical reasoning benchmarks.
- Explore integrating reasoning-aware adapters into the fine-tuning pipeline.
