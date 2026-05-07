PYTHON := /Volumes/T7 Shield/ML/conda_envs/rag-web/bin/python
PROJECT_DIR := /Users/kamruzzamankhanalve/Documents/RAG research/web_retrieval/rag-web

.PHONY: ollama ingest query test clean help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  ollama   Start the Ollama server (run in a separate terminal)"
	@echo "  ingest   Scrape URLs, embed, and store in ChromaDB (run once)"
	@echo "  query    Start the interactive query REPL"
	@echo "  test     Run pipeline sanity check (no Ollama needed)"
	@echo "  clean    Delete the ChromaDB vector store"

ollama:
	ollama serve

ingest:
	cd "$(PROJECT_DIR)" && "$(PYTHON)" ingest.py

query:
	cd "$(PROJECT_DIR)" && "$(PYTHON)" query.py

test:
	cd "$(PROJECT_DIR)" && "$(PYTHON)" test_pipeline.py

clean:
	rm -rf "$(PROJECT_DIR)/chroma_db"
	@echo "Vector store deleted. Run 'make ingest' to rebuild."
