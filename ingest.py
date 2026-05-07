"""
ingest.py — Scrape URLs, chunk, embed, and persist to ChromaDB.
Run: /Volumes/T7\ Shield/ML/conda_envs/rag-web/bin/python ingest.py
"""

import chromadb
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def build_index() -> VectorStoreIndex:
    # ── 1. Embedding model only — no LLM needed during ingest ─────────────────
    print(f"[1/5] Loading embedding model '{config.EMBED_MODEL}' ...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    Settings.llm = None
    print("      Embedding model ready.")

    # ── 2. Scrape URLs ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Scraping {len(config.URLS)} URL(s):")
    for url in config.URLS:
        print(f"      • {url}")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=config.URLS)
    total_chars = sum(len(d.text) for d in documents)
    print(f"      Fetched {len(documents)} document(s)  ({total_chars:,} chars total).")

    # ── 3. Chunk ───────────────────────────────────────────────────────────────
    print(f"\n[3/5] Splitting into chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) ...")
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    print(f"      {len(nodes)} chunks created.")

    # ── 4. ChromaDB setup ──────────────────────────────────────────────────────
    print(f"\n[4/5] Connecting to ChromaDB at '{config.CHROMA_PATH}' ...")
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    print(f"      Collection '{config.COLLECTION_NAME}' ready "
          f"({collection.count()} existing vectors).")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # ── 5. Embed and persist ───────────────────────────────────────────────────
    print(f"\n[5/5] Embedding {len(nodes)} chunks and writing to ChromaDB ...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_ctx,
        show_progress=True,
    )
    print(f"\nDone. {collection.count()} vectors stored in "
          f"'{config.CHROMA_PATH}' / collection '{config.COLLECTION_NAME}'.")
    return index


if __name__ == "__main__":
    build_index()
