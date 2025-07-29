# query.py
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set same embedding model as used in ingestion
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Load GROQ_API_KEY from .env
load_dotenv()

# Load index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# Set up Groq LLM
llm = Groq(
    model="deepseek-r1-distill-llama-70b",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Create query engine
query_engine = index.as_query_engine(llm=llm)

# Take query from user
user_query = input("❓ Enter your question: ")
response = query_engine.query(user_query)

# Display result
print("\n🔍 Response:\n", response)
