# ── Data sources ──────────────────────────────────────────────────────────────
URLS = [
    "https://www.therundown.ai/",
    "https://huggingface.co/blog",
    "https://simonwillison.net",
]

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ── Models ─────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434"

# ── Vector store ───────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_web"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K = 5
