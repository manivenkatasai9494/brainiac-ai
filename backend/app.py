import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the frontend

# Load environment variables from .env file (ensure .env is in the rag_project root)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- RAG Setup (Load the vector store and create the chain) ---
# Path to the FAISS index relative to where app.py is executed
# Assuming app.py is in 'rag_project/backend' and 'vectorstore' is in 'rag_project'
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectorstore')

def load_retriever():
    """
    Loads the FAISS vector store and returns a retriever.
    """
    print("Loading FAISS index...")
    model_name = "all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    print("Retriever loaded successfully.")
    return retriever

def create_rag_chain(retriever):
    """
    Creates and returns the RAG chain using LangChain components.
    """
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return rag_chain

# Global variable to store the RAG chain, initialized once
rag_chain = None

# --- API Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_question():
    """
    API endpoint to receive questions from the frontend and return answers.
    """
    if rag_chain is None:
        return jsonify({"error": "Chatbot service is not ready. Please check server logs."}), 500

    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"Received question: '{question}'")
    try:
        start_time = time.time()
        answer = rag_chain.invoke(question)
        end_time = time.time()
        
        print(f"Answer generated in {end_time - start_time:.2f} seconds.")
        print(f"Answer: {answer}")
        
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"An error occurred while invoking the RAG chain: {e}")
        return jsonify({"error": "An internal error occurred while processing your request. Please try again."}), 500

# --- Main entry point to run the server ---
if __name__ == '__main__':
    # Initialize the RAG chain before starting the server
    try:
        retriever = load_retriever()
        rag_chain = create_rag_chain(retriever)
        print("RAG chain initialized and ready to serve requests.")
    except Exception as e:
        print(f"CRITICAL ERROR: RAG chain initialization failed: {e}")
        rag_chain = None

    print("Starting Flask server...")
    app.run(debug=True, port=5000)