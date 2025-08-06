# app.py

import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from retriever_loader import load_retriever
from rag_chain_builder import create_rag_chain

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global chain variable
rag_chain = None

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

# Run app
if __name__ == '__main__':
    try:
        retriever = load_retriever()
        rag_chain = create_rag_chain(retriever)
        print("RAG chain initialized and ready to serve requests.")
    except Exception as e:
        print(f"CRITICAL ERROR: RAG chain initialization failed: {e}")
        rag_chain = None

    print("Starting Flask server...")
    app.run(debug=True, port=5000)
