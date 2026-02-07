"""Vector store module for document embedding and retrieval."""

from typing import List
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Ensure environment variables are loaded
load_dotenv()

class VectorStore:
    """Manages vector store operations."""

    def __init__(self):
        """Initialize vector store with Google / Gemini embeddings."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document], k: int = 4, progress_callback=None):
        """
        Create vector store from documents with rate limiting (batching).

        Args:
            documents: List of documents to embed.
            k: Default number of documents to retrieve for a query.
            progress_callback: Optional function(msg, percent_complete) to report progress.
        """
        import time
        
        # Rate limit settings for free tier
        batch_size = 10
        delay = 10  # seconds

        if not documents:
            return

        total_docs = len(documents)
        total_batches = ((total_docs - 1) // batch_size) + 1
        
        print(f"Starting embedding of {total_docs} documents in batches of {batch_size}...")

        # Process first batch to initialize
        first_batch = documents[:batch_size]
        msg = f"Embedding batch 1/{total_batches}..."
        print(msg)
        if progress_callback:
            progress_callback(msg, 0.0)
            
        self.vectorstore = FAISS.from_documents(first_batch, self.embedding)
        
        # Process remaining batches
        for i in range(batch_size, total_docs, batch_size):
            current_batch_num = (i // batch_size) + 1
            
            # Wait for rate limits
            wait_msg = f"Waiting {delay}s to respect rate limits..."
            print(wait_msg)
            if progress_callback:
                 # Update progress but keep percentage roughly same
                progress_callback(wait_msg, (current_batch_num - 1) / total_batches)
            time.sleep(delay)
            
            batch = documents[i : i + batch_size]
            msg = f"Embedding batch {current_batch_num}/{total_batches}..."
            print(msg)
            if progress_callback:
                progress_callback(msg, current_batch_num / total_batches)
                
            self.vectorstore.add_documents(batch)

        if progress_callback:
            progress_callback("Finalizing index...", 1.0)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_retriever(self):
        """
        Get the retriever instance.

        Returns:
            Retriever instance.
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def save_index(self, folder_path: str):
        """
        Save the FAISS index to disk.

        Args:
            folder_path: Directory path to save the index.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is empty. Nothing to save.")
        self.vectorstore.save_local(folder_path)
        print(f"Vector store saved to {folder_path}")

    def load_index(self, folder_path: str, k: int = 4):
        """
        Load the FAISS index from disk.

        Args:
            folder_path: Directory path where the index is saved.
            k: Default number of documents to retrieve.
        """
        if not os.path.exists(folder_path):
             raise ValueError(f"Index path {folder_path} does not exist.")
        
        self.vectorstore = FAISS.load_local(
            folder_path, 
            self.embedding,
            allow_dangerous_deserialization=True # Safe since we created it
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"Vector store loaded from {folder_path}")

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query.
            k: Number of documents to retrieve.

        Returns:
            List of relevant documents.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_index first.")

        # If user passes a different k, build a temporary retriever with that k
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)
