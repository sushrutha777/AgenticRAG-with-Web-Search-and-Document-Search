"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Model Configuration
    # syntax: "provider:model_name"
    LLM_MODEL = "google_genai:gemini-2.5-flash"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # LangChain's Google integration looks for 'GOOGLE_API_KEY' by default.
        # We map your GEMINI_API_KEY to it here.
        os.environ["GOOGLE_API_KEY"] = cls.GEMINI_API_KEY
        
        # 'temperature=0' is recommended for RAG to reduce hallucinations
        return init_chat_model(cls.LLM_MODEL, temperature=0)
