"""
RAG pipeline using LangChain Expression Language (LCEL) with Gemini
"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline using LCEL with Gemini"""
    
    def __init__(self, retriever: BaseRetriever, llm = None):
        self.retriever = retriever
        
        if llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model=config.LLM_MODEL,
                google_api_key=config.get_google_api_key(),
                temperature=config.TEMPERATURE,
                max_output_tokens=config.MAX_OUTPUT_TOKENS,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                convert_system_message_to_human=True
            )
        else:
            self.llm = llm
        
        # Build the RAG chain
        self.rag_chain = self._build_rag_chain()
        logger.info("RAG pipeline initialized with Gemini")
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt"""
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', f'Document {i+1}')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            
            formatted.append(
                f"DOCUMENT {i+1} [Source: {source}, Page: {page}]:\n"
                f"{content}\n"
            )
        return "\n".join(formatted)
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template optimized for Gemini"""
        
        system_template = """You are an expert assistant that answers questions based ONLY on the provided documents.

CRITICAL INSTRUCTIONS:
1. Your knowledge is limited to the documents below. Do not use any external knowledge.
2. If the answer cannot be found in the documents, say: "I cannot find this information in the provided documents."
3. Always cite your sources using [Source: filename, Page: X] format.
4. Be concise, accurate, and helpful.
5. If asked about documents or sources, list the relevant ones.

DOCUMENTS:
{context}

QUESTION: {question}

Please provide a helpful answer based on the documents:"""
        
        prompt = ChatPromptTemplate.from_template(system_template)
        return prompt
    
    def _build_rag_chain(self):
        """Build the RAG chain using LCEL"""
        
        # Define the retrieval and formatting chain
        retrieve_and_format = RunnableParallel({
            "context": lambda x: self._format_documents(self.retriever.get_relevant_documents(x["question"])),
            "question": lambda x: x["question"]
        })
        
        # Create the prompt
        prompt = self._create_prompt()
        
        # Build the full chain
        chain = (
            retrieve_and_format
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def invoke(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """Invoke the RAG chain with a question"""
        logger.info(f"Processing question: '{question}'")
        
        try:
            # Get documents for logging and response
            docs = self.retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(docs)} documents")
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"QUERY: {question}")
                print(f"RETRIEVED {len(docs)} DOCUMENTS:")
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    print(f"{i+1}. {source} (Page: {page})")
                    print(f"   Preview: {doc.page_content[:150]}...")
                print(f"{'='*60}")
            
            # Get answer
            result = self.rag_chain.invoke({"question": question})
            
            # Extract sources
            sources = [
                {
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "content_preview": doc.page_content[:150] + "..."
                }
                for doc in docs
            ]
            
            logger.info(f"Answer generated successfully")
            
            return {
                "question": question,
                "answer": result,
                "sources": sources,
                "num_sources": len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "error": str(e)
            }
    
    def batch_invoke(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions"""
        results = []
        for question in questions:
            results.append(self.invoke(question))
        return results
    
    def stream(self, question: str):
        """Stream the response"""
        logger.info(f"Streaming response for question: '{question}'")
        
        # Get documents first
        docs = self.retriever.get_relevant_documents(question)
        logger.info(f"Retrieved {len(docs)} documents")
        
        # Create chain for streaming
        retrieve_and_format = RunnableParallel({
            "context": lambda x: self._format_documents(docs),
            "question": lambda x: x["question"]
        })
        
        prompt = self._create_prompt()
        chain = retrieve_and_format | prompt | self.llm
        
        # Stream the response
        for chunk in chain.stream({"question": question}):
            yield chunk.content if hasattr(chunk, 'content') else str(chunk)

if __name__ == "__main__":
    # Test the RAG pipeline
    from vectorstore import VectorStoreManager
    
    # Initialize components
    manager = VectorStoreManager()
    retriever = manager.get_advanced_retriever(strategy="basic")
    
    # Create RAG pipeline
    rag = RAGPipeline(retriever)
    
    # Test query
    test_result = rag.invoke("What is this document about?", verbose=True)
    print(f"\nANSWER:\n{test_result['answer']}")