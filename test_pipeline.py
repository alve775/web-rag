"""
test_pipeline.py — Sanity-check the embed + retrieval pipeline without Ollama.

What it does:
  1. Scrapes the first URL in config.URLS
  2. Chunks it and prints the first 3 chunks with token counts + metadata
  3. Embeds + indexes into an in-memory (ephemeral) ChromaDB collection
  4. Retrieves top-3 chunks for "latest LLM models" and prints scores
  5. Exits 0 (PASS) or 1 (FAIL)

Run: /Volumes/T7\ Shield/ML/conda_envs/rag-web/bin/python test_pipeline.py
"""

import sys
import traceback

import chromadb
import tiktoken
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

TOKENIZER = tiktoken.get_encoding("cl100k_base")
TEST_QUERY = "latest LLM models"
TOP_K = 3
DIVIDER = "─" * 60


def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text))


def _header(step: str) -> None:
    print(f"\n{DIVIDER}\n{step}\n{DIVIDER}")


def run() -> None:
    # ── Step 1: embedding model (no LLM) ──────────────────────────────────────
    _header("STEP 1 — Load embedding model")
    print(f"Model : {config.EMBED_MODEL}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    Settings.llm = None
    print("Status: OK")

    # ── Step 2: scrape first URL only ─────────────────────────────────────────
    url = config.URLS[0]
    _header(f"STEP 2 — Scrape first URL")
    print(f"URL   : {url}")
    docs = BeautifulSoupWebReader().load_data(urls=[url])
    assert len(docs) > 0, "No documents returned from scraper"
    total_chars = sum(len(d.text) for d in docs)
    print(f"Docs  : {len(docs)}  |  Total chars: {total_chars:,}")
    print("Status: OK")

    # ── Step 3: chunk and inspect first 3 nodes ───────────────────────────────
    _header("STEP 3 — Chunk documents  (showing first 3)")
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(docs)
    assert len(nodes) > 0, "Splitter produced no chunks"
    print(f"Total chunks : {len(nodes)}")

    for i, node in enumerate(nodes[:3], 1):
        toks = token_count(node.text)
        src = node.metadata.get("URL") or node.metadata.get("url") or "—"
        preview = node.text[:120].replace("\n", " ").strip()
        print(f"\n  Chunk {i}")
        print(f"    tokens   : {toks}")
        print(f"    source   : {src}")
        print(f"    preview  : {preview!r}")
    print("\nStatus: OK")

    # ── Step 4: build ephemeral in-memory index ───────────────────────────────
    _header("STEP 4 — Build in-memory index (no disk write)")
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("test_pipeline")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_ctx, show_progress=True)
    print(f"Vectors stored : {collection.count()}")
    print("Status: OK")

    # ── Step 5: retrieve top-3 without LLM ────────────────────────────────────
    _header(f"STEP 5 — Vector retrieval  query='{TEST_QUERY}'  top_k={TOP_K}")
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    results = retriever.retrieve(TEST_QUERY)
    assert len(results) > 0, "Retriever returned no results"

    for i, node in enumerate(results, 1):
        score = node.score if node.score is not None else 0.0
        src = node.node.metadata.get("URL") or node.node.metadata.get("url") or "—"
        preview = node.node.text[:120].replace("\n", " ").strip()
        print(f"\n  Result {i}")
        print(f"    score    : {score:.4f}")
        print(f"    source   : {src}")
        print(f"    preview  : {preview!r}")
    print("\nStatus: OK")


def main() -> None:
    print(f"\n{'=' * 60}")
    print("  rag-web  —  pipeline sanity check  (no Ollama)")
    print(f"{'=' * 60}")
    try:
        run()
        print(f"\n{'=' * 60}")
        print("  RESULT: PASS")
        print(f"{'=' * 60}\n")
        sys.exit(0)
    except Exception as exc:
        print(f"\n{'=' * 60}")
        print(f"  RESULT: FAIL")
        print(f"  Reason: {exc}")
        print(f"{'=' * 60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
