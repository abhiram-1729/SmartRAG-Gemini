"""
Vector store management and retrieval strategies for Gemini
"""
import logging
from typing import List, Tuple, Optional, Union

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations and advanced retrieval strategies"""
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = config.VECTOR_STORE_PATH
        
        # Initialize embeddings based on configuration
        if config.use_gemini_embeddings():
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.get_google_api_key(),
                task_type="retrieval_document"
            )
            logger.info("Using Gemini embeddings for vector store")
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Using HuggingFace embeddings: {config.EMBEDDING_MODEL}")
        
        # Load existing vector store
        try:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME
            )
            count = self.vector_store._collection.count()
            logger.info(f"Vector store loaded from {persist_directory}")
            logger.info(f"Collection contains {count} vectors")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def _get_llm(self, temperature: float = None):
        """Get Gemini LLM instance"""
        if temperature is None:
            temperature = config.TEMPERATURE
        
        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.get_google_api_key(),
            temperature=temperature,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            convert_system_message_to_human=True
        )
    
    def get_basic_retriever(self, k: int = None, search_type: str = "similarity") -> BaseRetriever:
        """Get basic similarity retriever"""
        if k is None:
            k = config.RETRIEVAL_K
        
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        logger.info(f"Created basic retriever with search_type={search_type}, k={k}")
        return retriever
    
    def get_multi_query_retriever(self, llm = None, k: int = None) -> BaseRetriever:
        """Get multi-query retriever for diverse perspectives"""
        if k is None:
            k = config.RETRIEVAL_K
        
        if llm is None:
            llm = self._get_llm(temperature=0)
        
        base_retriever = self.get_basic_retriever(k=k*2)  # Get more docs for diversity
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            include_original=True
        )
        
        logger.info(f"Created multi-query retriever with k={k}")
        return retriever
    
    def get_contextual_compression_retriever(self, llm = None, k: int = None) -> BaseRetriever:
        """Get retriever with contextual compression"""
        if k is None:
            k = config.RETRIEVAL_K
        
        if llm is None:
            llm = self._get_llm(temperature=0)
        
        base_retriever = self.get_basic_retriever(k=k*2)
        
        # Create compressor
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create compression retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        logger.info(f"Created contextual compression retriever with k={k}")
        return retriever
    
    def get_hybrid_retriever(self, documents: List[Document] = None, k: int = None) -> BaseRetriever:
        """Get hybrid retriever combining semantic and keyword search"""
        if k is None:
            k = config.RETRIEVAL_K
        
        # Get documents if not provided
        if documents is None:
            documents_data = self.vector_store.get()
            documents = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in zip(
                    documents_data['documents'],
                    documents_data['metadatas']
                )
            ]
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        
        # Create vector retriever
        vector_retriever = self.get_basic_retriever(k=k)
        
        # Combine retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]  # Equal weighting
        )
        
        logger.info(f"Created hybrid retriever with k={k}")
        return ensemble_retriever
    
    def get_advanced_retriever(self, 
                              strategy: str = "basic",
                              k: int = None) -> BaseRetriever:
        """Get retriever based on specified strategy"""
        strategies = {
            "basic": self.get_basic_retriever,
            "multi_query": self.get_multi_query_retriever,
            "compression": self.get_contextual_compression_retriever,
            "hybrid": lambda k: self.get_hybrid_retriever(k=k),
        }
        
        if strategy not in strategies:
            logger.warning(f"Strategy {strategy} not found. Using basic retriever.")
            strategy = "basic"
        
        logger.info(f"Using retrieval strategy: {strategy}")
        
        if strategy in ["multi_query", "compression"]:
            return strategies[strategy](None, k)
        else:
            return strategies[strategy](k)
    
    def search_documents(self, query: str, k: int = None) -> List[Document]:
        """Direct document search for debugging"""
        if k is None:
            k = config.RETRIEVAL_K
        
        docs = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(docs)} documents for query: '{query}'")
        
        for i, doc in enumerate(docs):
            logger.info(f"Doc {i+1} - Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"Content preview: {doc.page_content[:100]}...")
        
        return docs
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        if k is None:
            k = config.RETRIEVAL_K
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Found {len(results)} documents with scores for query: '{query}'")
        
        for i, (doc, score) in enumerate(results):
            logger.info(f"Doc {i+1} - Score: {score:.4f} - Source: {doc.metadata.get('source', 'Unknown')}")
        
        return results

if __name__ == "__main__":
    # Test vector store
    manager = VectorStoreManager()
    test_query = "What is this document about?"
    docs = manager.search_documents(test_query)