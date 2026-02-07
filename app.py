"""app.py — RAG Search with left-side history only (handles non-string answers)"""
import os

# Gemini to use v1 (not v1beta)
os.environ["GOOGLE_GENAI_API_VERSION"] = "v1"

# Silence USER_AGENT warning
os.environ["USER_AGENT"] = "CollegeRAG/1.0 (student project)"

import streamlit as st
from pathlib import Path
import sys
import time
from datetime import datetime
import json

# Ensure the repo src is importable
sys.path.append(str(Path(__file__).parent))

# Your RAG components
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

st.set_page_config(page_title="RAG Search", layout="centered")


def init_session_state():
    """Initialize session state keys (do this before creating widgets)."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "searching" not in st.session_state:
        st.session_state.searching = False
    # history: list of dicts {id, question, answer, retrieved_docs, time, created_at}
    if "history" not in st.session_state:
        st.session_state.history = []
    # selected history index in original order (None if nothing selected)
    if "selected_history_index" not in st.session_state:
        st.session_state.selected_history_index = None


# Path to save the FAISS index
VECTORSTORE_PATH = "faiss_index"

@st.cache_resource
def initialize_rag(rebuild_index: bool = False):
    """
    Cached initialization of RAG components.
    
    Args:
        rebuild_index: If True, force re-embedding of documents.
    """
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    vector_store = VectorStore()
    
    # Check if index exists and we don't want to rebuild
    if os.path.exists(VECTORSTORE_PATH) and not rebuild_index:
        try:
            vector_store.load_index(VECTORSTORE_PATH)
            # We can't easily know the doc count without metadata on disk, 
            # so we'll just say "Loaded from disk"
            doc_count = "Loaded from disk"
        except Exception as e:
            st.warning(f"Failed to load existing index: {e}. Rebuilding...")
            rebuild_index = True
            
    # If we need to build (fresh start or rebuild requested)
    if not os.path.exists(VECTORSTORE_PATH) or rebuild_index:
        # Combine URLs from config and local files from 'data' folder
        sources = Config.DEFAULT_URLS
        if os.path.exists("data"):
            sources.append("data")
            
        documents = doc_processor.process_urls(sources)
        
        # Progress bar UI
        progress_bar = st.progress(0, text="Starting ingestion...")
        status_text = st.empty()
        
        def update_progress(msg, percent):
            # percent is 0.0 to 1.0
            p = min(max(percent, 0.0), 1.0)
            progress_bar.progress(p, text=msg)

        try:
            vector_store.create_vectorstore(documents, progress_callback=update_progress)
            vector_store.save_index(VECTORSTORE_PATH)
            doc_count = len(documents)
            
            progress_bar.progress(1.0, text="Ingestion complete!")
            time.sleep(1) # Let user see 100%
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            raise e

    graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
    graph_builder.build()
    return graph_builder, doc_count


def normalize_answer(ans) -> str:
    """
    Convert various types of 'answer' that the RAG/LLM might return to a single string.
    - list -> join elements with double newline
    - dict -> pretty JSON
    - bytes -> decode
    - other scalars -> str()
    """
    if ans is None:
        return ""
    if isinstance(ans, str):
        return ans
    if isinstance(ans, (list, tuple)):
        try:
            if all(isinstance(x, str) for x in ans):
                return "\n\n".join(ans)
            return "\n\n".join(
                json.dumps(x, ensure_ascii=False)
                if isinstance(x, (dict, list))
                else str(x)
                for x in ans
            )
        except Exception:
            return str(ans)
    if isinstance(ans, dict):
        try:
            return json.dumps(ans, indent=2, ensure_ascii=False)
        except Exception:
            return str(ans)
    if isinstance(ans, (bytes, bytearray)):
        try:
            return ans.decode("utf-8", errors="replace")
        except Exception:
            return str(ans)
    return str(ans)


def render_chat_view(hist_index: int | None):
    """Render the selected conversation (chat-like) in the main area."""
    if hist_index is None:
        st.info("No conversation selected — ask a question to start.")
        return

    if hist_index < 0 or hist_index >= len(st.session_state.history):
        st.error("Selected conversation not found.")
        return

    item = st.session_state.history[hist_index]
    st.markdown(
        f"### Conversation — {item['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.markdown(f"**Q:** {item['question']}")
    st.markdown("**A:**")
    st.code(item["answer"], language="")
    st.caption(f" Response time: {item.get('time', 0):.2f}s")

    with st.expander("Source Documents"):
        docs = item.get("retrieved_docs", [])
        if not docs:
            st.write("No source documents saved for this conversation.")
        else:
            for i, doc in enumerate(docs, 1):
                content = getattr(doc, "page_content", str(doc))
                preview = content[:1200] + ("..." if len(content) > 1200 else "")
                st.text_area(
                    f"Document {i}", value=preview, height=160, disabled=True
                )


def main():
    init_session_state()

    st.title("Agentic RAG: Docs, Web & Wiki")
    st.markdown("Ask questions about your **Documents**, **Wikipedia**, or **Web Search**.")

    # Initialize RAG once (cached)
    if not st.session_state.initialized:
        with st.spinner("Loading RAG system..."):
            try:
                rag_system, num_chunks = initialize_rag()
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"System ready! ({num_chunks} document chunks loaded)")
            except Exception as e:
                st.session_state.initialized = False
                st.error(f"Failed to initialize RAG: {e}")
                return

    st.markdown("---")

    # Sidebar: HISTORY ONLY
    with st.sidebar:
        st.markdown("## History")
        if st.session_state.history:
            labels = []
            for idx, item in enumerate(reversed(st.session_state.history)):
                original_id = len(st.session_state.history) - idx  # 1-based
                label = (
                    f"{original_id}. "
                    f"{item['question'][:60]}{'...' if len(item['question']) > 60 else ''}"
                )
                labels.append(label)

            choice = st.selectbox(
                "Select a conversation", options=["(none)"] + labels, index=0
            )

            if choice != "(none)":
                num_str = choice.split(".", 1)[0]
                try:
                    orig_idx = int(num_str) - 1  # zero-based
                    st.session_state.selected_history_index = orig_idx
                except Exception:
                    st.session_state.selected_history_index = None
            else:
                st.session_state.selected_history_index = None

            if st.button("Clear history"):
                st.session_state.history = []
                st.session_state.selected_history_index = None
                st.rerun()
        else:
            st.info("No history yet. Ask something!")

        st.markdown("---")
        if st.button("Rebuild Index"):
            st.session_state.initialized = False
            st.session_state.rag_system = None
            # Clearing cache resource to force re-run of initialize_rag
            st.cache_resource.clear()
            # We set a flag in session state if we want to pass it, 
            # but since we cleared cache, the next call will run.
            # However, initialize_rag needs to know to rebuild.
            # Since we can't easily pass args to cached function without changing cache key every time,
            # we can rely on clearing the cache and manually deleting the folder or 
            # simpler: Just use a flag here to tell the main loop to call with rebuild=True
            if os.path.exists(VECTORSTORE_PATH):
                import shutil
                try:
                    shutil.rmtree(VECTORSTORE_PATH)
                    st.success("Index deleted. App will rebuild on reload.")
                except Exception as e:
                    st.error(f"Error deleting index: {e}")
            st.rerun()

    # Main area: show selected chat if any
    selected_idx = st.session_state.get("selected_history_index", None)
    if selected_idx is not None:
        render_chat_view(selected_idx)
        st.markdown("---")

    # Search Form (always shown) — disabled while searching
    disabled = st.session_state.searching

    with st.form(key="search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?",
            key="question_input",
            disabled=disabled,
        )
        submit = st.form_submit_button("Search", disabled=disabled)

    # Handle submit
    if submit:
        q = (question or "").strip()
        if not q:
            st.warning("Please enter a question before searching.")
        elif not st.session_state.initialized or st.session_state.rag_system is None:
            st.error("RAG system is not ready.")
        else:
            st.session_state.searching = True

            with st.spinner("Searching..."):
                start_time = time.time()
                try:
                    result = st.session_state.rag_system.run(q)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    result = None
                elapsed_time = time.time() - start_time

            if result:
                raw_answer = result.get("answer")
                normalized_answer = normalize_answer(raw_answer)

                st.markdown("### Answer")
                st.code(normalized_answer, language="")
                st.caption(f"Response time: {elapsed_time:.2f} seconds")

                with st.expander("Source Documents"):
                    for i, doc in enumerate(result.get("retrieved_docs", []), 1):
                        content = getattr(doc, "page_content", str(doc))
                        preview = content[:1200] + (
                            "..." if len(content) > 1200 else ""
                        )
                        st.text_area(
                            f"Document {i}", value=preview, height=160, disabled=True
                        )

                entry = {
                    "id": len(st.session_state.history) + 1,
                    "question": q,
                    "answer": normalized_answer,
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "time": elapsed_time,
                    "created_at": datetime.now(),
                }
                st.session_state.history.append(entry)
                st.session_state.selected_history_index = entry["id"] - 1

            st.session_state.searching = False


if __name__ == "__main__":
    main()
