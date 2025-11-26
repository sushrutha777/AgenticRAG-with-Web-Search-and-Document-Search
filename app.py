"""Streamlit UI for Agentic RAG System - Fixed version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS for full-width button
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""  # stored text_input value

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    # This is cached across reruns until source changes
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    vector_store = VectorStore()

    urls = Config.DEFAULT_URLS
    documents = doc_processor.process_urls(urls)
    vector_store.create_vectorstore(documents)

    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm
    )
    graph_builder.build()

    return graph_builder, len(documents)

def main():
    init_session_state()

    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    # Initialize system if not already done
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            try:
                rag_system, num_chunks = initialize_rag()
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.session_state.initialized = False

    st.markdown("---")

    # Put the search form (single instance) with explicit key and explicit text_input key
    # The text is stored in st.session_state.question_input so reruns don't recreate multiple controls
    with st.form(key="search_form"):
        st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?",
            key="question_input"
        )
        submit = st.form_submit_button("🔍 Search")

    # Process search when form is submitted
    if submit:
        question = st.session_state.get("question_input", "").strip()
        if not question:
            st.warning("Please enter a question before searching.")
        elif not st.session_state.initialized or st.session_state.rag_system is None:
            st.error("RAG system is not ready. Please wait for initialization to complete.")
        else:
            with st.spinner("Searching..."):
                start_time = time.time()
                try:
                    result = st.session_state.rag_system.run(question)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    result = None

                elapsed_time = time.time() - start_time

                if result:
                    # Add to history
                    st.session_state.history.append({
                        'question': question,
                        'answer': result.get('answer', 'No answer returned'),
                        'time': elapsed_time
                    })

                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.success(result.get('answer', 'No answer returned'))

                    # Show retrieved docs in expander
                    with st.expander("📄 Source Documents"):
                        for i, doc in enumerate(result.get('retrieved_docs', []), 1):
                            content = getattr(doc, "page_content", str(doc))
                            preview = content[:300] + ("..." if len(content) > 300 else "")
                            st.text_area(
                                f"Document {i}",
                                value=preview,
                                height=100,
                                disabled=True
                            )

                    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # Show recent history (last 3)
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}{'...' if len(item['answer'])>200 else ''}")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()
