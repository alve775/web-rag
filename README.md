# rag-web

> A fully local RAG (Retrieval-Augmented Generation) pipeline that scrapes live web pages, indexes them with vector embeddings, and answers questions using a locally-running LLM — no API keys, no cloud.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange)
![Ollama](https://img.shields.io/badge/Ollama-local-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it does

1. **Scrapes** a configurable list of URLs using `BeautifulSoupWebReader`
2. **Chunks** the text into overlapping segments with `SentenceSplitter`
3. **Embeds** each chunk locally using `BAAI/bge-small-en-v1.5` (HuggingFace)
4. **Stores** vectors persistently in ChromaDB on disk
5. **Answers** questions at query time by retrieving the top-K chunks and passing them to a local LLM via Ollama

---

## Stack

| Layer | Tool |
| --- | --- |
| Orchestration | [LlamaIndex](https://github.com/run-llama/llama_index) 0.14 |
| Web scraping | `BeautifulSoupWebReader` |
| Chunking | `SentenceSplitter` |
| Embeddings | HuggingFace [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| Vector store | [ChromaDB](https://www.trychroma.com/) — persistent, on-disk |
| LLM | [Ollama](https://ollama.com) — runs fully locally |
| Config | `config.py` — all parameters in one place |

---

## Prerequisites

### Python 3.10+

### Ollama

```bash
# macOS
brew install ollama

# Start the server (keep running in a separate terminal)
ollama serve

# Pull a model — use any model you have locally
ollama pull gemma4:e4b   # or: qwen3.5:9b, llama3.2:3b, etc.
```

Verify:

```bash
ollama list              # should show your pulled models
```

---

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd rag-web

# Create and activate conda environment
conda create -n rag-web python=3.11
conda activate rag-web

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1 — Ingest

Scrapes all configured URLs, chunks the text, embeds it, and stores vectors in `./chroma_db`. Run once, or again after changing URLs or config.

```bash
make ingest
```

```text
[1/5] Loading embedding model 'BAAI/bge-small-en-v1.5' ...
      Embedding model ready.
[2/5] Scraping 3 URL(s):
      • https://www.therundown.ai/
      • https://huggingface.co/blog
      • https://simonwillison.net
      Fetched 3 document(s)  (84,321 chars total).
[3/5] Splitting into chunks (size=512, overlap=64) ...
      187 chunks created.
[4/5] Connecting to ChromaDB at './chroma_db' ...
      Collection 'rag_web' ready (0 existing vectors).
[5/5] Embedding 187 chunks and writing to ChromaDB ...
Done. 187 vectors stored in './chroma_db' / collection 'rag_web'.
```

### 2 — Query

Loads the persisted index and starts an interactive Q&A loop. No re-embedding.

```bash
make query
```

```text
Q: What are the latest AI models released on HuggingFace?

A: Several new models have recently been released on HuggingFace, including ...

Sources:
  [1] score=0.8732  https://huggingface.co/blog/...
  [2] score=0.8401  https://huggingface.co/blog/...
  [3] score=0.7955  https://simonwillison.net/...
```

Type `exit` or `quit` to stop. Press `Ctrl-C` at any time.

### 3 — Test (no Ollama required)

Runs a quick sanity check on the embed + retrieval pipeline without touching the LLM.

```bash
make test
```

### 4 — Refresh the index

```bash
make clean    # deletes chroma_db/
make ingest   # rebuilds from scratch
```

---

## Make targets

```bash
make ollama   # start the Ollama server (separate terminal)
make ingest   # scrape → chunk → embed → store
make query    # load index → interactive REPL
make test     # pipeline check, no LLM needed
make clean    # wipe chroma_db/
```

---

## Data flow

```text
┌─────────────────────────────────────────────────────────┐
│                        INGEST                           │
│                                                         │
│  URLs ──► BeautifulSoupWebReader ──► raw text           │
│                    │                                    │
│                    ▼                                    │
│           SentenceSplitter ──► overlapping chunks       │
│                    │                                    │
│                    ▼                                    │
│        HuggingFaceEmbedding ──► 384-dim vectors         │
│                    │                                    │
│                    ▼                                    │
│         ChromaDB (./chroma_db) ──► persisted            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                        QUERY                            │
│                                                         │
│  Question ──► HuggingFaceEmbedding ──► query vector     │
│                    │                                    │
│                    ▼                                    │
│    ChromaDB similarity search ──► top-K chunks          │
│                    │                                    │
│                    ▼                                    │
│    Ollama LLM (context + question) ──► answer           │
└─────────────────────────────────────────────────────────┘
```

---

## Configuration

All parameters are in [`config.py`](config.py). Edit and re-run `make ingest` to apply.

| Parameter | Default | What changing it does |
| --- | --- | --- |
| `URLS` | 3 sites | Add/remove URLs to expand or narrow the knowledge base |
| `CHUNK_SIZE` | `512` | Smaller = more granular retrieval; larger = more context per chunk |
| `CHUNK_OVERLAP` | `64` | Higher overlap reduces information loss at chunk boundaries |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Swap for a larger model (e.g. `BAAI/bge-large-en-v1.5`) for better recall |
| `LLM_MODEL` | `gemma4:e4b` | Any model available via `ollama list` |
| `OLLAMA_URL` | `http://localhost:11434` | Change if Ollama runs on a remote host |
| `CHROMA_PATH` | `./chroma_db` | Directory where vectors are persisted |
| `COLLECTION_NAME` | `rag_web` | ChromaDB namespace — change to isolate ingestion runs |
| `TOP_K` | `5` | Chunks retrieved per query — increase for broader context |

> **Note:** Changing `EMBED_MODEL` or `CHUNK_SIZE` invalidates existing vectors. Run `make clean && make ingest` after either change.

---

## Project structure

```text
rag-web/
├── config.py          # all tuneable parameters
├── ingest.py          # scrape → chunk → embed → store
├── query.py           # load index → REPL → answer + sources
├── test_pipeline.py   # sanity check without Ollama
├── requirements.txt
├── Makefile
└── README.md
```

---

## Next steps

- **Re-ranking** — apply a cross-encoder after retrieval to re-score chunks before sending to the LLM
- **Hybrid search** — combine dense vector search with BM25 sparse retrieval using `QueryFusionRetriever`
- **Metadata filtering** — tag chunks by source domain and filter at query time
- **LangChain port** — see [`../rag-web-langchain`](../rag-web-langchain) for an equivalent implementation using `WebBaseLoader`, `RecursiveCharacterTextSplitter`, `ChatOllama`, and `RetrievalQA`

---

## License

MIT
