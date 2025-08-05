import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Define file paths
DATA_PATH = "data/Updated_major_dataset.txt"
FAISS_INDEX_PATH = "vectorstore"

def create_vector_store():
    """
    Loads a document, splits it into chunks, and creates a FAISS vector store.
    """
    print("Loading document...")
    loader = TextLoader(DATA_PATH, encoding="utf8")
    document = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(document)

    print(f"Created {len(text_chunks)} chunks.")

    print("Initializing HuggingFace embeddings model...")
    model_name = "all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index saved successfully.")

if __name__ == "__main__":
    create_vector_store()