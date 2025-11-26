# End To End RAG Document Search

This project is a **Multi-source Agentic RAG system** built with **Streamlit,LangChain,** and **LangGraph.** It enables users to chat with their data regardless of the format. It supports **PDFs, Text files, and Website URLs** providing **AI-powered answers** with accurate context retrieval using **Google Gemini 2.5 models.**

## 🚀 Features
- **Universal Data Support:** Seamlessly ingest and process PDF documents, Text files (.txt),    and Website URLs.
- **Agentic Workflow:** Powered by LangChain and LangGraph to orchestrate retrieval, reasoning, and answer generation.
- **Google Gemini Integration:** Utilizes Gemini 2.5 Flash/Pro for fast and accurate reasoning.
- **Smart Embeddings:** Uses Google Generative AI Embeddings for semantic search across different data formats.
- **Interactive Streamlit UI:** A unified interface to manage data sources and chat in real-time.
- **Blazing Fast Setup:** Uses uv package manager for ultra-fast dependency resolution and installation.

## 📦 Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/sushrutha777/RAG.git
   cd End-To-End-RAG-Search
2. Install uv(If not installed):
   ```bash
   pip install uv
3. Create Virtual Environment:
   ```bash
    # Create virtual environment
    uv venv
    # Activate the virtual environment
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
4. Install the required dependencies:
   ```bash
    uv add -r requirements.txt
4. Create a .env file in the project root and add your Google Gemini API key:
   ```bash
    GEMINI_API_KEY=your_api_key_here
5. Run the Streamlit app:
   ```bash
    streamlit run app.py
