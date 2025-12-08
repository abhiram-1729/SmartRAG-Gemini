"""
Configuration settings for the RAG application with Gemini
"""
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model configurations
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Can also use Gemini embeddings
    LLM_MODEL = "gemini-2.5-flash"  # Using Gemini 1.5 Flash (better quota)
    # For embeddings: "models/embedding-001" (when using Gemini embeddings)
    
    # Vector store settings
    VECTOR_STORE_PATH = "./vector_store"
    CHROMA_COLLECTION_NAME = "rag_documents_gemini"
    
    # Retrieval settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 4
    SIMILARITY_THRESHOLD = 0.7
    
    # Document paths
    DOCUMENT_PATHS = [
        "./data/documents/",
        "./data/sample_document_1.txt",
        "./data/sample_document_2.md"
    ]
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.docx']
    
    # Generation settings
    TEMPERATURE = 0.3
    MAX_OUTPUT_TOKENS = 1000
    TOP_P = 0.95
    TOP_K = 40
    
    @classmethod
    def get_google_api_key(cls) -> str:
        """Get Google API key from environment"""
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return key
    
    @classmethod
    def use_gemini_embeddings(cls) -> bool:
        """Whether to use Gemini embeddings (requires separate embedding model)"""
        return os.getenv("USE_GEMINI_EMBEDDINGS", "false").lower() == "true"

config = Config()


# Streamlit UI settings
class StreamlitConfig:
    # UI settings
    PAGE_TITLE = "RAG Chat Assistant"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    # Chat settings
    MAX_CHAT_HISTORY = 50
    SHOW_SOURCES = True
    SHOW_SIMILARITY_SCORES = True
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB = 100
    ALLOWED_FILE_TYPES = ['.txt', '.md', '.pdf', '.docx']
    
    # Analytics settings
    TRACK_QUERY_METRICS = True
    PLOT_RESPONSE_TIMES = True

# Add to existing Config class or create a new instance
streamlit_config = StreamlitConfig()