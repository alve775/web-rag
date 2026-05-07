"""
query.py — Interactive REPL against the persisted ChromaDB RAG index.
Run: /Volumes/T7\ Shield/ML/conda_envs/rag-web/bin/python query.py
"""

import chromadb
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def load_index() -> VectorStoreIndex:
    print(f"[1/3] Loading embedding model '{config.EMBED_MODEL}' ...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)

    print(f"[2/3] Connecting to Ollama ({config.LLM_MODEL}) ...")
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_URL,
        request_timeout=120.0,
        thinking=False,
    )

    print(f"[3/3] Loading index from ChromaDB at '{config.CHROMA_PATH}' ...")
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    vector_count = collection.count()
    if vector_count == 0:
        raise RuntimeError(
            f"Collection '{config.COLLECTION_NAME}' is empty. "
            "Run ingest.py first."
        )
    print(f"      Found {vector_count} vectors in '{config.COLLECTION_NAME}'.")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_ctx
    )
    return index


def run_repl(index: VectorStoreIndex) -> None:
    engine = index.as_query_engine(similarity_top_k=config.TOP_K)

    print(f"\nReady. Ask anything (top_k={config.TOP_K}, llm={config.LLM_MODEL}).")
    print('Type "exit" or "quit" to stop.\n')
    print("─" * 60)

    while True:
        try:
            question = input("\nQ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        response = engine.query(question)

        print(f"\nA: {response}\n")

        if response.source_nodes:
            print("Sources:")
            for i, node in enumerate(response.source_nodes, 1):
                score = node.score if node.score is not None else 0.0
                url = node.node.metadata.get("URL") or node.node.metadata.get("url") or "unknown"
                print(f"  [{i}] score={score:.4f}  {url}")

        print("─" * 60)


if __name__ == "__main__":
    index = load_index()
    run_repl(index)
