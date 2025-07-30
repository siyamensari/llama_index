import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.prompts import PromptTemplate
import nltk

# ✅ Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# ✅ Load environment variables
load_dotenv()

# ✅ Set embedding model and LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))

# ✅ Load index from persistent storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index_ids = [index_struct.index_id for index_struct in storage_context.index_store.index_structs()]
index = load_index_from_storage(storage_context, index_id=index_ids[0])

# ✅ Custom prompt allowing prior knowledge
prompt_template = PromptTemplate(
    "You're a helpful assistant. Use both the context below and your own general knowledge to answer.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# ✅ Initialize chat engine with history and context
chat_engine = CondensePlusContextChatEngine.from_defaults(
    llm=llm,
    retriever=index.as_retriever(),
    text_qa_template=prompt_template,
    verbose=False,
)

# ✅ CLI loop
print("🤖 AI Chat Ready. Type your message or 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break

    try:
        response = chat_engine.chat(user_input)
        print(f"AI: {response.response}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")
