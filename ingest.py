"""
Document ingestion pipeline with Gemini-compatible embeddings
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIngestor:
    """Handles document loading, splitting, and embedding"""
    
    def __init__(self):
        # Choose embedding model based on configuration
        if config.use_gemini_embeddings():
            logger.info("Using Gemini embeddings")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.get_google_api_key(),
                task_type="retrieval_document"
            )
        else:
            logger.info(f"Using HuggingFace embeddings: {config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self, paths: Optional[List[str]] = None) -> List:
        """Load documents from specified paths"""
        if paths is None:
            paths = config.DOCUMENT_PATHS
        
        all_documents = []
        
        for path in paths:
            path_obj = Path(path)
            
            if not path_obj.exists():
                logger.warning(f"Path does not exist: {path}")
                continue
            
            if path_obj.is_dir():
                # Load all supported files from directory
                for ext in config.SUPPORTED_EXTENSIONS:
                    loader = DirectoryLoader(
                        str(path),
                        glob=f"**/*{ext}",
                        loader_cls=self._get_loader_for_extension(ext)
                    )
                    try:
                        documents = loader.load()
                        all_documents.extend(documents)
                        logger.info(f"Loaded {len(documents)} documents from {path} with extension {ext}")
                    except Exception as e:
                        logger.error(f"Error loading {ext} files from {path}: {e}")
            else:
                # Load single file
                loader = self._get_loader_for_file(str(path))
                if loader:
                    try:
                        documents = loader.load()
                        all_documents.extend(documents)
                        logger.info(f"Loaded {len(documents)} documents from {path}")
                    except Exception as e:
                        logger.error(f"Error loading {path}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def _get_loader_for_file(self, file_path: str):
        """Get appropriate loader based on file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None
    
    def _get_loader_for_extension(self, ext: str):
        """Get loader class for extension"""
        mapping = {
            '.pdf': PyPDFLoader,
            '.txt': lambda path: TextLoader(path, encoding='utf-8'),
            '.md': UnstructuredMarkdownLoader,
        }
        return mapping.get(ext.lower())
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List, persist_directory: str = None) -> Chroma:
        """Create and persist vector store"""
        if persist_directory is None:
            persist_directory = config.VECTOR_STORE_PATH
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=config.CHROMA_COLLECTION_NAME
        )
        
        # Persist to disk
        vector_store.persist()
        logger.info(f"Vector store created and persisted to {persist_directory}")
        logger.info(f"Collection: {config.CHROMA_COLLECTION_NAME}, Contains: {vector_store._collection.count()} vectors")
        
        return vector_store
    
    def ingest(self, paths: Optional[List[str]] = None) -> Chroma:
        """Complete ingestion pipeline"""
        logger.info("Starting document ingestion pipeline...")
        
        # Step 1: Load documents
        documents = self.load_documents(paths)
        if not documents:
            raise ValueError("No documents loaded. Check your document paths.")
        
        # Step 2: Split documents
        chunks = self.split_documents(documents)
        
        # Step 3: Create vector store
        vector_store = self.create_vector_store(chunks)
        
        logger.info("Ingestion pipeline completed successfully!")
        return vector_store

if __name__ == "__main__":
    # Example usage
    ingestor = DocumentIngestor()
    vector_store = ingestor.ingest()