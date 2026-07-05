import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PDF_FILE = "ENT 202 Introduction to Entrepreneurial Ventures_1.pdf"
DB_DIR = "chroma_db"

print("Parsing PDF...")
loader = PyPDFLoader(PDF_FILE)
pages = loader.load()

print("Splitting text into segments...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

print("Connecting to OpenAI and saving database locally...")
# This will consume credit once. Push it to disk.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

print(f"Success! Your database is built inside the '{DB_DIR}' folder.")
