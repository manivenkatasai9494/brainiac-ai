
### ****Enterprise AI Knowledge Companion with Smart Search****

This project is a full-stack, production-ready RAG (Retrieval-Augmented Generation) pipeline designed to function as an intelligent AI companion for searching and interacting with internal company documents. It demonstrates core skills in building agentic AI systems and leveraging LLMs to deliver scalable, real-world solutions.

-----

### **Project Goal**

The primary goal of this project is to build a smart assistant that allows employees to search and interact with internal company documents (HR manuals, policies, etc.) using natural language queries. The system retrieves the most relevant information and generates accurate, context-aware responses using an LLM, simulating a production-ready conversational AI agent.

-----

### **Core Features & Architecture**

#### 1\. **Production-Grade RAG Pipeline**

  * **Semantic Search & LLM-Based QA**: The system accepts natural language questions, intelligently retrieves the most relevant document chunks, and synthesizes a concise answer using a large language model. This demonstrates a practical application of a production-ready RAG system.
  * **Vector Database**: Documents are parsed, chunked, and their embeddings are stored in a **FAISS vector store** for efficient semantic search and retrieval.
  * **Agentic Backend**: A **Flask backend** serves as a FastAPI-like endpoint, exposing the core agent functionality. This allows the frontend to communicate with the RAG pipeline via a clean, standardized API.

#### 2\. **Document Processing & Management**

  * **Document Ingestion**: The system processes a plain text document (`Updated_major_dataset.txt`), chunks it, and generates vector embeddings. This showcases experience with data preprocessing for AI systems.
  * **Retriever Optimization**: The RAG chain is designed to retrieve the most relevant information based on the user's query, a critical component of advanced RAG strategies.

#### 3\. **User Interface & Experience**

  * **Conversational Interface**: A simple yet effective frontend, built with **HTML, CSS, and JavaScript**, provides a real-time conversational interface for users to interact with the RAG system.
  * **API Integration**: The frontend is integrated with the backend's `/ask` endpoint, demonstrating a full-stack development capability and an understanding of API communication.

-----

### **Technologies & Frameworks**

  * **Python**: Core programming language.
  * **LangChain**: Framework for building the RAG pipeline.
  * **Flask**: Backend web framework for the API endpoint.
  * **LangChain-Google-GenAI**: Integration for using Google's Gemini models.
  * **FAISS**: Vector database for efficient similarity search.
  * **`sentence-transformers`**: Library for generating document embeddings.
  * **HTML/CSS/JavaScript**: Frontend development for the user interface.
  * **Python-dotenv**: For managing and securing API keys.

-----

### **How to Run the Project**

This project is designed to be easily set up and run. Follow these steps to get the full-stack application working.

#### **Step 1: Environment Setup**

1.  Navigate to the project's root directory.
2.  Create a Conda environment with a compatible Python version (3.11+).
    ```bash
    conda create --name rag_project python=3.11
    conda activate rag_project
    ```
3.  Install all project dependencies using the `requirements.txt` file in the root directory.
    ```bash
    pip install -r requirements.txt
    ```

#### **Step 2: Backend & Vector Store Preparation**

1.  Create a `.env` file in the root directory and add your Google API key.
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
2.  Run the vector store creation script. This will process the `Updated_major_dataset.txt` file and generate the embeddings.
    ```bash
    python create_vectorstore.py
    ```
3.  Navigate to the `backend` folder and start the Flask server.
    ```bash
    cd backend
    python app.py
    ```

#### **Step 3: Frontend Access**

1.  Open the `index.html` file located in the `frontend` directory in your web browser.
2.  The chatbot interface will load and connect to the running backend. You can now start asking questions about the employee handbook.

-----

### **Future Enhancements & Advanced Concepts**

This project lays the foundation for a more complex and scalable system. Planned future work includes:

  * **Contextual Memory**: Implementing a memory system to handle multi-turn, context-aware conversations.
  * **Role-Based Access Control**: Designing and integrating a JWT-based authentication system to serve different documents and features based on user roles.
  * **Document Summarization**: Adding a feature to generate concise summaries of documents or policies.
  * **Multi-Modal Support**: Expanding the system to handle documents with images and tables using tools like `pytesseract` or multi-modal LLMs.

