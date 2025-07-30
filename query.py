import os, json
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


nltk.data.path.append("./nltk_data")


# Environment and models
load_dotenv()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))

# Load storage and available index IDs
storage_dir = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=storage_dir)

# ✅ Fix: Call index_structs() and extract index_id from each object
index_ids = [index_struct.index_id for index_struct in storage_context.index_store.index_structs()]

# Get user query
user_query = input("❓ Enter your question: ")

# Keyword extraction
def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return set([word for word in tokens if word.isalnum() and word not in stop_words])

query_keywords = extract_keywords(user_query)

# Loop through indexes, query each, score responses
response_scores = []
for idx_id in index_ids:
    try:
        index = load_index_from_storage(storage_context, index_id=idx_id)
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(user_query)
        response_text = response.response.lower()

        # Match score = count of keyword overlaps
        match_score = sum(1 for kw in query_keywords if kw in response_text)
        response_scores.append((match_score, idx_id, response))
    except Exception as e:
        print(f"[Error querying index '{idx_id}']: {e}")

# Sort and show best response
response_scores.sort(reverse=True, key=lambda x: x[0])

if response_scores and response_scores[0][0] > 0:
    top_score, top_id, best_response = response_scores[0]
    print(f"\n🎯 Best match from index '{top_id}' (score={top_score}):\n{best_response}")
else:
    print("\n⚠️ No relevant keyword matches found in any index.")
