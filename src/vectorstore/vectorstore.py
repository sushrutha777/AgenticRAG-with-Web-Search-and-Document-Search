"""Vector store module for document embedding and retrieval."""

from typing import List
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
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
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document], k: int = 4):
        """
        Create vector store from documents.

        Args:
            documents: List of documents to embed.
            k: Default number of documents to retrieve for a query.
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
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
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")

        # If user passes a different k, build a temporary retriever with that k
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)
