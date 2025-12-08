"""
Main script to run the RAG application with Gemini
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ingest import DocumentIngestor
from vectorstore import VectorStoreManager
from rag_chain import RAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ingest_documents(paths: Optional[list] = None, force_recreate: bool = False) -> bool:
    """Run the document ingestion pipeline"""
    logger.info("Starting document ingestion...")
    try:
        ingestor = DocumentIngestor()
        vector_store = ingestor.ingest(paths)
        logger.info("Ingestion completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return False

def setup_rag(retrieval_strategy: str = "basic") -> Optional[RAGPipeline]:
    """Set up and return RAG pipeline"""
    logger.info(f"Setting up RAG pipeline with strategy: {retrieval_strategy}")
    
    try:
        # Load vector store
        manager = VectorStoreManager()
        
        # Get retriever based on strategy
        retriever = manager.get_advanced_retriever(strategy=retrieval_strategy)
        
        # Create RAG pipeline
        rag = RAGPipeline(retriever)
        
        logger.info("RAG pipeline setup completed!")
        return rag
    except Exception as e:
        logger.error(f"Failed to setup RAG pipeline: {e}")
        return None

def interactive_mode(rag: RAGPipeline):
    """Run interactive Q&A mode"""
    logger.info("Entering interactive mode. Type 'quit', 'exit', or 'q' to end.")
    print("\n" + "="*60)
    print("RAG Application with Gemini - Interactive Mode")
    print("="*60)
    print("Type your questions below. The system will retrieve relevant")
    print("documents and generate answers based on them.")
    print("="*60)
    
    while True:
        try:
            question = input("\n📝 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Process question with verbose output
            result = rag.invoke(question, verbose=True)
            
            # Display answer
            print("\n" + "="*60)
            print("🤖 ANSWER:")
            print("="*60)
            print(result["answer"])
            
            # Display sources
            if result["sources"]:
                print(f"\n📚 Sources used ({result['num_sources']}):")
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. {source['source']} (Page: {source['page']})")
                    print(f"   Preview: {source['content_preview']}")
            
            print("="*60)
            
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"An error occurred: {e}")

def stream_mode(rag: RAGPipeline, question: str):
    """Stream the response for a single question"""
    print(f"\n🔍 Question: {question}")
    print("-" * 60)
    
    # Get documents first
    from vectorstore import VectorStoreManager
    manager = VectorStoreManager()
    docs = manager.search_documents(question, k=4)
    
    if docs:
        print(f"📄 Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"   {i}. {source}")
    
    print("\n💭 Generating answer...\n")
    
    # Stream the response
    print("🤖 Answer: ", end="", flush=True)
    for chunk in rag.stream(question):
        print(chunk, end="", flush=True)
    
    print("\n" + "="*60)

def answer_query(question: str, retrieval_strategy: str = "basic", 
                 stream: bool = False) -> dict:
    """Answer a single query"""
    rag = setup_rag(retrieval_strategy)
    if rag:
        if stream:
            stream_mode(rag, question)
            return {"status": "streamed"}
        else:
            result = rag.invoke(question, verbose=True)
            return result
    else:
        return {"answer": "Failed to setup RAG pipeline", "sources": []}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Application using LangChain with Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ingest
  %(prog)s query -q "What is RAG?"
  %(prog)s query -q "Explain the features" -s multi_query --stream
  %(prog)s interactive -s hybrid
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["ingest", "query", "interactive"],
        help="Mode: ingest documents, answer a query, or interactive mode"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to answer (required for query mode)"
    )
    
    parser.add_argument(
        "--paths", "-p",
        nargs="+",
        help="Document paths to ingest (optional)"
    )
    
    parser.add_argument(
        "--strategy", "-s",
        choices=["basic", "multi_query", "compression", "hybrid"],
        default="basic",
        help="Retrieval strategy (default: basic)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response (query mode only)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate vector store (ingest mode only)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Execute based on mode
    if args.mode == "ingest":
        success = ingest_documents(args.paths, args.force)
        sys.exit(0 if success else 1)
    
    elif args.mode == "query":
        if not args.query:
            parser.error("--query is required for query mode")
        
        logger.info(f"Processing query: {args.query}")
        result = answer_query(args.query, args.strategy, args.stream)
        
        if not args.stream and 'answer' in result:
            print(f"\n📝 Question: {result['question']}")
            print(f"\n🤖 Answer: {result['answer']}")
            
            if result.get('sources'):
                print(f"\n📚 Sources used ({result['num_sources']}):")
                for source in result['sources']:
                    print(f"- {source['source']} (Page: {source['page']})")
    
    elif args.mode == "interactive":
        rag = setup_rag(args.strategy)
        if rag:
            interactive_mode(rag)
        else:
            logger.error("Failed to setup RAG pipeline")
            sys.exit(1)

if __name__ == "__main__":
    main()