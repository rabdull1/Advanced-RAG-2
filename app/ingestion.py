"""Document ingestion and chunking for RAG pipeline."""

from pathlib import Path
from typing import List, Dict
import json
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from app.config import config


class DocumentIngestor:
    """Handles document loading, chunking, and embedding storage."""
    
    def __init__(self):
        """Initialize the document ingestor with embedding model and vector DB."""
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.client = chromadb.PersistentClient(path=str(config.vector_db_dir))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def load_documents(self, file_path: Path) -> List[Dict]:
        """Load documents from a JSON or PDF file."""
        if file_path.suffix.lower() == '.pdf':
            return self.load_pdf(file_path)
        elif file_path.suffix.lower() == '.json':
            return self.load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def load_pdf(self, file_path: Path) -> List[Dict]:
        """Load text from a PDF file."""
        reader = PdfReader(file_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return [{"id": file_path.stem, "text": text}]
    
    def load_json(self, file_path: Path) -> List[Dict]:
        """Load documents from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("Unsupported JSON format")
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        text_length = len(text)
        
        for i in range(0, text_length, config.chunk_size - config.chunk_overlap):
            chunk_text = text[i:i + config.chunk_size]
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "doc_id": doc_id,
                "metadata": {"chunk_index": len(chunks)}
            })
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def store_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Store chunks and embeddings in vector database."""
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["metadata"]["chunk_index"]
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def ingest_file(self, file_path: Path, doc_id: str):
        """Complete ingestion pipeline for a single file."""
        documents = self.load_documents(file_path)
        
        for doc in documents:
            text = doc.get("text", "") or doc.get("content", "")
            if not text:
                continue
            
            chunks = self.chunk_text(text, doc_id)
            embeddings = self.generate_embeddings(chunks)
            self.store_chunks(chunks, embeddings)
        
        print(f"Successfully ingested {file_path}")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
