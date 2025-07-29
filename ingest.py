# ingest.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

# Load data from ./data
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# Load local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Create index with local embeddings
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Save index
index.storage_context.persist(persist_dir="./storage")

print("✅ Ingestion completed and index saved to './storage'")
