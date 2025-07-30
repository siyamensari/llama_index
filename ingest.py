import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from typing import List

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

pdf_dir = "./data/pdf"
docx_dir = "./data/docx"
text_dir = "./data/text"

def load_all_documents() -> List[Document]:
    documents = []

    if os.path.exists(text_dir):
        documents.extend(SimpleDirectoryReader(text_dir).load_data())

    if os.path.exists(pdf_dir):
        pdf_reader = PDFReader()
        for file in os.listdir(pdf_dir):
            if file.endswith(".pdf"):
                documents.extend(pdf_reader.load_data(os.path.join(pdf_dir, file)))

    if os.path.exists(docx_dir):
        docx_reader = DocxReader()
        for file in os.listdir(docx_dir):
            if file.endswith(".docx"):
                documents.extend(docx_reader.load_data(os.path.join(docx_dir, file)))

    return documents

def load_urls_from_file(file_path="urls.txt") -> List[Document]:
    if not os.path.exists(file_path):
        print("⚠️  No urls.txt file found.")
        return []

    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    if not urls:
        print("⚠️  No URLs found in urls.txt.")
        return []

    web_reader = SimpleWebPageReader()
    return web_reader.load_data(urls)

documents = load_all_documents() + load_urls_from_file()

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
index.storage_context.persist()

print("✅ Ingestion complete.")
