"""Configuration settings for the RAG pipeline."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration for RAG pipeline components."""
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    vector_db_dir: Path = base_dir / "data" / "vector_db"
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Generation settings
    model_name: str = "gemini-3.1-pro-preview"
    temperature: float = 0.7
    max_tokens: int = 500
    
    # API Keys
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    def __post_init__(self):
        """Create necessary directories."""
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


config = Config()
