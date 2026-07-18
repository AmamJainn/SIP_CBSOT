# Research Paper Intelligence System

**Author:** Amam Jain

An end-to-end NLP pipeline that lets you **search, summarize, extract keywords from, and compare** machine learning research papers using natural language queries — powered by semantic search, transformer-based summarization, and an LLM agent.

---

## Overview

This project turns a corpus of ML research papers into a queryable, conversational knowledge base. Instead of manually skimming abstracts, you can ask questions like *"find papers on vision transformers"* or *"compare CNNs and ViTs"* and get relevant, summarized answers automatically.

It combines classic NLP/IR techniques (embeddings + vector search) with modern generative AI (an LLM agent that decides which tool to call), all wrapped into a single, self-contained Jupyter notebook.

## Features

- **Semantic Search** — retrieves papers by meaning, not just keyword overlap, using Sentence-Transformer embeddings and a FAISS vector index
- **Automatic Summarization** — condenses paper abstracts with a BART model, scaling summary length to the input
- **Keyword / Keyphrase Extraction** — surfaces the most representative terms from any text using KeyBERT
- **Named Entity Recognition** — pulls out models, datasets, and institutions mentioned in a paper
- **Paper Comparison** — compares two papers side by side across objective, methodology, contributions, advantages, limitations, and applications
- **Conversational Agent** — exposes all of the above as tools to an LLM (Llama 3.1 8B via Groq) through LangChain, so it can be queried in plain English

## How It Works (Pipeline)

1. **Load & clean the dataset** — pulls the `CShorten/ML-ArXiv-Papers` dataset from Hugging Face (titles + abstracts of ML papers from arXiv), drops nulls/duplicates, and merges title + abstract into one text field.
2. **Generate embeddings** — encodes each paper with `all-MiniLM-L6-v2`, producing 384-dimensional semantic vectors.
3. **Build a FAISS index** — indexes the (L2-normalized) embeddings with `IndexFlatIP` for fast cosine-similarity search; embeddings and index are cached to disk after the first run.
4. **Summarize** — condenses retrieved abstracts with `facebook/bart-large-cnn`.
5. **Extract keywords** — pulls top keyphrases with KeyBERT.
6. **Named entity recognition** — a general-purpose NER pipeline tags models, datasets, and institutions.
7. **Retrieval quality check** — sanity-checks search results against a handful of hand-picked test queries.
8. **LLM agent layer** — wraps search+summarize, keyword extraction, and paper comparison as LangChain tools and hands them to a Llama 3.1 agent (via Groq).
9. **Demo queries** — end-to-end examples of the agent picking the right tool for a given question.

## Tech Stack

| Purpose | Library |
|---|---|
| Semantic embeddings | `sentence-transformers` |
| Vector search | `faiss-cpu` |
| Summarization & NER | `transformers` |
| Keyword extraction | `keybert` |
| Agent framework | `langchain`, `langchain-community`, `langchain-core`, `langchain-huggingface` |
| LLM inference | `langchain-groq` (Llama 3.1 8B) |
| Dataset loading | `datasets` |
| Data handling | `pandas`, `numpy`, `scikit-learn` |

## Dataset

[`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) on Hugging Face — titles and abstracts of machine learning papers from arXiv. The notebook works with a configurable sample (default: 15,000 papers) to keep the demo lightweight; the same pipeline scales to the full dataset given enough compute and storage.

## Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com) (only needed for the LLM agent section)
- Enough RAM to hold the sentence-transformer and BART models in memory
- A GPU is recommended for faster embedding/summarization, but the pipeline runs on CPU as well

## Installation

```bash
pip install datasets sentence-transformers faiss-cpu keybert \
    transformers==4.46.3 huggingface_hub==0.26.2 tokenizers==0.20.3 sentence-transformers==3.3.1 \
    langchain langchain-community langchain-core langchain-huggingface langchain-groq
```

## Setup

The LLM agent needs a Groq API key. Set it as an environment variable before running those cells:

```bash
export GROQ_API_KEY="your-api-key-here"
```

If you're running in Google Colab, the notebook will instead try to read `GROQ_API_KEY` from Colab's secrets manager automatically.

## Usage

1. Open `Research_Paper_Intelligence_System_1.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the cells top to bottom — later sections depend on variables/functions created earlier (dataset `df`, `embed_model`, `index`, etc.).
3. Once the index is built, search and summarize directly:

   ```python
   display_search_and_summarize("deep learning for medical image analysis", k=5)
   ```

4. Or, once the agent is set up, ask it questions in natural language:

   ```python
   agent.invoke({
       "messages": [
           {"role": "user", "content": "Find the top 3 papers on reinforcement learning and summarize them."}
       ]
   })
   ```

## Example Queries

- "Find the top 3 research papers on Vision Transformers and summarize them."
- "Extract the top 5 keywords from: Deep Learning for Medical Image Reconstruction."
- "Compare a paper about Vision Transformers with a paper about Convolutional Neural Networks."

## Project Structure

```
.
├── Research_Paper_Intelligence_System_1.ipynb   # main notebook — full pipeline
├── paper_embeddings.npy                         # cached embeddings (created on first run)
└── paper_faiss.index                            # cached FAISS index (created on first run)
```

## Possible Extensions

- Swap in a domain-specific embedding model (e.g., SPECTER2) for stronger academic-paper retrieval
- Move to an approximate FAISS index (IVF, HNSW) to scale to larger corpora
- Add metadata filtering by year, category, or venue
- Add a lightweight FastAPI or Streamlit front end for interactive use

## Author

**Amam Jain**
