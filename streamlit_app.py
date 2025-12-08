"""
Streamlit UI for the RAG Application with Gemini
"""
import streamlit as st
import os
import sys
import logging
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import your existing modules
try:
    from config import Config, config
    from ingest import DocumentIngestor
    from vectorstore import VectorStoreManager
    from rag_chain import RAGPipeline
    from langchain_core.documents import Document
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all dependencies are installed.")
    st.stop()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    
    .chat-message.assistant {
        background-color: #475063;
        color: white;
    }
    
    .chat-message .avatar {
        width: 20%;
    }
    
    .chat-message .message {
        width: 80%;
    }
    
    .source-card {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    
    .metric-card {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class RAGStreamlitApp:
    """Streamlit application for RAG with Gemini"""
    
    def __init__(self):
        self.rag_pipeline = None
        self.vector_store_manager = None
        self.initialized = False
        
        # Initialize session state
        self._init_session_state()

        # Track whether we already attempted auto-init to avoid loops
        if 'auto_init_attempted' not in st.session_state:
            st.session_state.auto_init_attempted = False

        # Restore pipeline objects across reruns
        self._restore_pipeline_from_session()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        if 'vector_store_initialized' not in st.session_state:
            st.session_state.vector_store_initialized = False
        
        if 'retrieval_stats' not in st.session_state:
            st.session_state.retrieval_stats = {
                'total_queries': 0,
                'avg_retrieved_docs': 0,
                'retrieval_times': []
            }
        
        if 'rag_config' not in st.session_state:
            st.session_state.rag_config = {
                'retrieval_strategy': 'basic',
                'k_value': 4,
                'temperature': 0.3,
                'use_gemini_embeddings': False
            }

        if 'rag_ready' not in st.session_state:
            st.session_state.rag_ready = False
    
    def _restore_pipeline_from_session(self):
        """Restore pipeline components stored in session_state after reruns"""
        if st.session_state.get('rag_ready'):
            self.vector_store_manager = st.session_state.get('vector_store_manager')
            self.rag_pipeline = st.session_state.get('rag_pipeline')
            if self.vector_store_manager and self.rag_pipeline:
                self.initialized = True
    
    def initialize_rag(self):
        """Initialize the RAG pipeline"""
        try:
            with st.spinner("Initializing RAG pipeline..."):
                # Load vector store
                self.vector_store_manager = VectorStoreManager()
                
                # Get retriever based on current configuration
                strategy = st.session_state.rag_config['retrieval_strategy']
                retriever = self.vector_store_manager.get_advanced_retriever(strategy=strategy)
                
                # Create RAG pipeline
                self.rag_pipeline = RAGPipeline(retriever)
                
                self.initialized = True
                st.session_state.rag_ready = True
                st.session_state.vector_store_manager = self.vector_store_manager
                st.session_state.rag_pipeline = self.rag_pipeline
                st.session_state.vector_store_initialized = True
                
                # Get vector store stats
                vector_store_info = self._get_vector_store_info()
                
                return True, f"✅ RAG pipeline initialized successfully! {vector_store_info}"
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False, f"❌ Failed to initialize RAG: {str(e)}"
    
    def _get_vector_store_info(self) -> str:
        """Get information about the vector store"""
        try:
            if self.vector_store_manager and hasattr(self.vector_store_manager, 'vector_store'):
                count = self.vector_store_manager.vector_store._collection.count()
                return f"Vector store contains {count} document chunks."
        except:
            pass
        return ""
    
    def _ensure_initialized(self):
        """Initialize RAG once if not already ready."""
        if self.initialized:
            return True
        if st.session_state.auto_init_attempted:
            return False

        st.session_state.auto_init_attempted = True
        success, message = self.initialize_rag()
        if success:
            st.success(message)
            return True
        st.warning(message)
        return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the RAG pipeline"""
        if not self.initialized or not self.rag_pipeline:
            # Attempt an automatic initialization before failing
            if not self._ensure_initialized():
                return {
                    'answer': "RAG pipeline not initialized. Please click 'Initialize RAG Pipeline' in the sidebar and try again.",
                    'sources': [],
                    'num_sources': 0
                }
        
        try:
            # Record start time
            import time
            start_time = time.time()
            
            # Process query
            result = self.rag_pipeline.invoke(query)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            st.session_state.retrieval_stats['total_queries'] += 1
            st.session_state.retrieval_stats['avg_retrieved_docs'] = (
                (st.session_state.retrieval_stats['avg_retrieved_docs'] * 
                 (st.session_state.retrieval_stats['total_queries'] - 1) + 
                 result['num_sources']) / 
                st.session_state.retrieval_stats['total_queries']
            )
            st.session_state.retrieval_stats['retrieval_times'].append(processing_time)
            
            # Add processing time to result
            result['processing_time'] = processing_time
            
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'processing_time': 0
            }
    
    def ingest_documents(self, uploaded_files: List) -> str:
        """Ingest uploaded documents"""
        try:
            if not uploaded_files:
                return "⚠️ No files uploaded."
            
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file to temporary directory
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Ingest documents
                ingestor = DocumentIngestor()
                ingestor.ingest(file_paths)
                
                # Reset RAG pipeline to use new documents
                self.initialized = False
                st.session_state.rag_ready = False
                st.session_state.vector_store_initialized = False
                
                # Store uploaded file names
                st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
                
                return f"✅ Successfully ingested {len(uploaded_files)} files: {', '.join([f.name for f in uploaded_files])}"
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            return f"❌ Error ingesting documents: {str(e)}"
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # RAG Configuration
            st.subheader("RAG Settings")
            
            # Retrieval Strategy
            strategy = st.selectbox(
                "Retrieval Strategy",
                ["basic", "multi_query", "compression", "hybrid"],
                index=["basic", "multi_query", "compression", "hybrid"].index(
                    st.session_state.rag_config['retrieval_strategy']
                ),
                help="Choose the retrieval strategy"
            )
            
            # Number of documents to retrieve (k)
            k_value = st.slider(
                "Number of documents to retrieve (k)",
                min_value=1,
                max_value=10,
                value=st.session_state.rag_config['k_value'],
                help="Number of document chunks to retrieve for each query"
            )
            
            # Temperature
            temperature = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.rag_config['temperature'],
                step=0.1,
                help="Higher values make output more creative, lower values more deterministic"
            )
            
            # Update config
            st.session_state.rag_config.update({
                'retrieval_strategy': strategy,
                'k_value': k_value,
                'temperature': temperature
            })
            
            # Update Config object
            config.RETRIEVAL_K = k_value
            config.TEMPERATURE = temperature
            
            st.divider()
            
            # Document Upload Section
            st.subheader("📄 Document Management")
            
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=['txt', 'md', 'pdf', 'docx'],
                accept_multiple_files=True,
                help="Upload text, markdown, or PDF files"
            )
            
            if uploaded_files:
                if st.button("🚀 Ingest Documents", type="primary", use_container_width=True):
                    with st.spinner("Processing documents..."):
                        result = self.ingest_documents(uploaded_files)
                        st.success(result)
                        
                        # Refresh the RAG pipeline
                        if st.session_state.get('vector_store_initialized', False):
                            success, message = self.initialize_rag()
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
            
            # Show uploaded files
            if st.session_state.uploaded_files:
                st.subheader("Uploaded Files")
                for file_name in st.session_state.uploaded_files[-5:]:  # Show last 5
                    st.markdown(f"📄 {file_name}")
            
            st.divider()
            
            # System Information
            st.subheader("ℹ️ System Info")
            
            if st.session_state.vector_store_initialized:
                st.success("✅ Vector store loaded")
                vector_info = self._get_vector_store_info()
                st.info(vector_info)
            else:
                st.warning("⚠️ Vector store not loaded")
            
            # Initialize button
            if not self.initialized:
                if st.button("🔧 Initialize RAG Pipeline", type="primary", use_container_width=True):
                    success, message = self.initialize_rag()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.success("✅ RAG pipeline ready!")
                
                # Reset chat button
                if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.rerun()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("🤖 RAG Chat Assistant")
        st.markdown("Ask questions about your documents using Google's Gemini model")
        
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show sources if available
                    if message.get("sources"):
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.markdown(f"""
                                **Source {i+1}:** {source['source']}
                                *Page:* {source['page']}
                                *Preview:* {source['content_preview']}
                                """)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Process query
                    result = self.process_query(prompt)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Display sources if available
                    if result['sources']:
                        with st.expander(f"📚 Sources ({result['num_sources']})"):
                            for i, source in enumerate(result['sources']):
                                st.markdown(f"""
                                **Source {i+1}:** {source['source']}
                                *Page:* {source['page']}
                                *Preview:* {source['content_preview']}
                                """)
                        
                        # Show similarity scores if available
                        if hasattr(self.vector_store_manager, 'similarity_search_with_score'):
                            try:
                                scores = self.vector_store_manager.similarity_search_with_score(
                                    prompt, 
                                    k=result['num_sources']
                                )
                                if scores:
                                    with st.expander("📊 Similarity Scores"):
                                        for i, (doc, score) in enumerate(scores):
                                            st.progress(
                                                float(score), 
                                                text=f"Source {i+1}: {score:.3f}"
                                            )
                            except:
                                pass
                    
                    # Show processing time
                    if 'processing_time' in result:
                        st.caption(f"⏱️ Processed in {result['processing_time']:.2f} seconds")
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources'],
                "num_sources": result['num_sources']
            })
            
            # Add to query history
            st.session_state.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": prompt,
                "answer": result['answer'],
                "num_sources": result['num_sources'],
                "processing_time": result.get('processing_time', 0)
            })
    
    def render_analytics_dashboard(self):
        """Render analytics dashboard"""
        st.title("📊 Analytics Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Queries",
                value=st.session_state.retrieval_stats['total_queries']
            )
        
        with col2:
            st.metric(
                label="Avg Docs Retrieved",
                value=f"{st.session_state.retrieval_stats['avg_retrieved_docs']:.1f}"
            )
        
        with col3:
            avg_time = (
                sum(st.session_state.retrieval_stats['retrieval_times']) / 
                max(1, len(st.session_state.retrieval_stats['retrieval_times']))
            )
            st.metric(
                label="Avg Response Time",
                value=f"{avg_time:.2f}s"
            )
        
        # Query history
        if st.session_state.chat_history:
            st.subheader("📝 Recent Queries")
            
            for i, query in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {query['question'][:50]}..."):
                    st.markdown(f"**Question:** {query['question']}")
                    st.markdown(f"**Answer:** {query['answer'][:200]}...")
                    st.markdown(f"**Sources:** {query['num_sources']} documents")
                    st.markdown(f"**Time:** {query['processing_time']:.2f}s")
        
        # Performance chart
        if len(st.session_state.retrieval_stats['retrieval_times']) > 1:
            st.subheader("⏱️ Response Time Trend")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(st.session_state.retrieval_stats['retrieval_times']))),
                y=st.session_state.retrieval_stats['retrieval_times'],
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#4CAF50')
            ))
            
            fig.update_layout(
                xaxis_title="Query Number",
                yaxis_title="Response Time (seconds)",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_document_explorer(self):
        """Render document explorer"""
        st.title("📚 Document Explorer")
        
        if not self.initialized or not self.vector_store_manager:
            if not self._ensure_initialized():
                st.warning("Please initialize the RAG pipeline first (sidebar).")
                return
        
        # Search documents
        search_query = st.text_input("Search in documents", placeholder="Enter keywords to search...")
        
        if search_query:
            with st.spinner("Searching documents..."):
                try:
                    # Perform search
                    docs = self.vector_store_manager.search_documents(search_query, k=6)
                    
                    if docs:
                        st.success(f"Found {len(docs)} relevant documents")
                        
                        for i, doc in enumerate(docs):
                            with st.expander(f"📄 Document {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                                st.markdown("**Content:**")
                                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                    else:
                        st.info("No documents found matching your search.")
                except Exception as e:
                    st.error(f"Error searching documents: {str(e)}")
        
        # Show document statistics
        st.subheader("📈 Document Statistics")
        
        try:
            # Get vector store information
            if hasattr(self.vector_store_manager, 'vector_store'):
                count = self.vector_store_manager.vector_store._collection.count()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Chunks", count)
                
                with col2:
                    st.metric("Retrieval Strategy", st.session_state.rag_config['retrieval_strategy'])
        except:
            st.info("Vector store statistics not available.")
    
    def run(self):
        """Run the Streamlit application"""
        # Sidebar
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "📚 Documents"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            self.render_document_explorer()
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            Powered by Google Gemini • Built with LangChain • UI with Streamlit
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main entry point for Streamlit app"""
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("""
        ⚠️ Google API key not found!
        
        Please create a `.env` file in the project directory with:
        ```
        GOOGLE_API_KEY=your_api_key_here
        ```
        
        Get your API key from: https://makersuite.google.com/app/apikey
        """)
        
        # Option to input API key directly
        api_key = st.text_input("Or enter your Google API key:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("API key set! Please restart the app.")
            st.stop()
        else:
            st.stop()
    
    # Initialize and run the app
    app = RAGStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()