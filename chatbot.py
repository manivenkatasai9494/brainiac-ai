import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define file paths
FAISS_INDEX_PATH = "vectorstore"

def load_retriever():
    """
    Loads the FAISS vector store from the local directory and returns a retriever.
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
    Creates and returns the RAG chain.
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

def main():
    """
    Main function to run the chatbot.
    """
    try:
        retriever = load_retriever()
        rag_chain = create_rag_chain(retriever)
        
        print("Chatbot is ready. Type 'exit' to quit.")
        while True:
            question = input("You: ")
            if question.lower() == 'exit':
                break
            
            answer = rag_chain.invoke(question)
            print(f"Bot: {answer}")
            
    except FileNotFoundError:
        print(f"Error: The FAISS index was not found at '{FAISS_INDEX_PATH}'.")
        print("Please run `python create_vectorstore.py` first to generate the index.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()