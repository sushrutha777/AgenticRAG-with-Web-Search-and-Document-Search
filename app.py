"""app.py — RAG Search with left-side history only (handles non-string answers)"""

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

st.set_page_config(page_title="RAG Search", page_icon="🔍", layout="centered")

# Small style tweaks
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    /* Slightly wider main column for chat-like feeling */
    .css-1d391kg { max-width: 980px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def init_session_state():
    """Initialize session state keys (do this before creating widgets)."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    # Widget-backed text value must exist before the widget is created
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
    if 'searching' not in st.session_state:
        st.session_state.searching = False
    # history: list of dicts {id, question, answer, retrieved_docs, time, created_at}
    if 'history' not in st.session_state:
        st.session_state.history = []
    # selected history index in original order (None if nothing selected)
    if 'selected_history_index' not in st.session_state:
        st.session_state.selected_history_index = None

@st.cache_resource
def initialize_rag():
    """Cached initialization of RAG components — runs once."""
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    vector_store = VectorStore()
    urls = Config.DEFAULT_URLS
    documents = doc_processor.process_urls(urls)
    vector_store.create_vectorstore(documents)
    graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
    graph_builder.build()
    return graph_builder, len(documents)

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
    # If the result is already a string, return it unchanged
    if isinstance(ans, str):
        return ans
    # If list or tuple: join the textual elements. For non-str elements, convert each to string.
    if isinstance(ans, (list, tuple)):
        try:
            # If list of strings:
            if all(isinstance(x, str) for x in ans):
                return "\n\n".join(ans)
            # Else stringify each element
            return "\n\n".join(json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x) for x in ans)
        except Exception:
            return str(ans)
    # If dict -> pretty JSON
    if isinstance(ans, dict):
        try:
            return json.dumps(ans, indent=2, ensure_ascii=False)
        except Exception:
            return str(ans)
    # bytes
    if isinstance(ans, (bytes, bytearray)):
        try:
            return ans.decode('utf-8', errors='replace')
        except Exception:
            return str(ans)
    # fallback for any other type
    return str(ans)

def render_chat_view(hist_index: int | None):
    """Render the selected conversation (chat-like) in the main area."""
    if hist_index is None:
        st.info("No conversation selected — ask a question to start.")
        return

    # Bounds check
    if hist_index < 0 or hist_index >= len(st.session_state.history):
        st.error("Selected conversation not found.")
        return

    item = st.session_state.history[hist_index]
    st.markdown(f"### Conversation — {item['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"**Q:** {item['question']}")
    st.markdown("**A:**")
    # Show the normalized answer (safe)
    st.code(item['answer'], language="")  # code block formats long text nicely
    st.caption(f"⏱️ Response time: {item.get('time', 0):.2f}s")

    with st.expander("📄 Source Documents"):
        docs = item.get('retrieved_docs', [])
        if not docs:
            st.write("No source documents saved for this conversation.")
        else:
            for i, doc in enumerate(docs, 1):
                content = getattr(doc, "page_content", str(doc))
                preview = content[:1200] + ("..." if len(content) > 1200 else "")
                st.text_area(f"Document {i}", value=preview, height=160, disabled=True)

def main():
    init_session_state()

    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    # Initialize RAG once (cached)
    if not st.session_state.initialized:
        with st.spinner("Loading RAG system..."):
            try:
                rag_system, num_chunks = initialize_rag()
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
            except Exception as e:
                st.session_state.initialized = False
                st.error(f"Failed to initialize RAG: {e}")
                return

    st.markdown("---")

    # -------------------
    # Sidebar: HISTORY ONLY
    # -------------------
    with st.sidebar:
        st.markdown("## 💬 History")
        if st.session_state.history:
            # Build labels newest-first, but store indices in original order
            labels = []
            for idx, item in enumerate(reversed(st.session_state.history)):
                # Compute original index (1-based id stored in history)
                original_id = len(st.session_state.history) - idx  # 1-based
                label = f"{original_id}. {item['question'][:60]}{'...' if len(item['question'])>60 else ''}"
                labels.append(label)

            # Default index is 0 -> "(none)"
            choice = st.selectbox("Select a conversation", options=["(none)"] + labels, index=0)

            if choice != "(none)":
                # Parse leading number to map back to original 0-based index in history
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
                st.experimental_rerun()
        else:
            st.info("No history yet. Ask something!")

    # -------------------
    # Main area: show selected chat if any
    # -------------------
    selected_idx = st.session_state.get("selected_history_index", None)
    if selected_idx is not None:
        render_chat_view(selected_idx)
        st.markdown("---")

    # -------------------
    # Search Form (always shown) — disabled while searching
    # -------------------
    disabled = st.session_state.searching

    with st.form(key="search_form"):
        # question_input key maps to st.session_state.question_input (created earlier)
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?",
            key="question_input",
            disabled=disabled
        )
        submit = st.form_submit_button("🔍 Search", disabled=disabled)

    # -------------------
    # Handle submit
    # -------------------
    if submit:
        q = (question or "").strip()
        if not q:
            st.warning("Please enter a question before searching.")
        elif not st.session_state.initialized or st.session_state.rag_system is None:
            st.error("RAG system is not ready.")
        else:
            # mark searching
            st.session_state.searching = True

            with st.spinner("Searching..."):
                start_time = time.time()
                try:
                    result = st.session_state.rag_system.run(q)
                except Exception as e:
                    # Show detailed error but don't crash the UI
                    st.error(f"Search failed: {e}")
                    result = None
                elapsed_time = time.time() - start_time

            if result:
                # Normalize answer to a string BEFORE storing or passing to any component that expects a string
                raw_answer = result.get("answer")
                normalized_answer = normalize_answer(raw_answer)

                # show immediate result in main area
                st.markdown("### 💡 Answer")
                st.code(normalized_answer, language="")
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result.get("retrieved_docs", []), 1):
                        content = getattr(doc, "page_content", str(doc))
                        preview = content[:1200] + ("..." if len(content) > 1200 else "")
                        st.text_area(f"Document {i}", value=preview, height=160, disabled=True)

                # append to history (store the normalized string answer)
                entry = {
                    "id": len(st.session_state.history) + 1,
                    "question": q,
                    "answer": normalized_answer,
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "time": elapsed_time,
                    "created_at": datetime.now()
                }
                st.session_state.history.append(entry)

                # auto-select the newly added conversation
                st.session_state.selected_history_index = entry["id"] - 1

            # done searching
            st.session_state.searching = False

if __name__ == "__main__":
    main()
