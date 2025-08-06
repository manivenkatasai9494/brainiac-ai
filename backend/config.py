# config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API key for LangChain
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Path to FAISS index
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectorstore')
