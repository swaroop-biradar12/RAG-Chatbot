import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = None

def load_document(filepath):
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath, encoding="utf-8")
    return loader.load()

def process_document(filepath):
    global vectorstore

    docs = load_document(filepath)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return len(chunks)

def retrieve_context(question, k=4):
    if vectorstore is None:
        return ""

    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

def clear_index():
    global vectorstore
    vectorstore = None
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)