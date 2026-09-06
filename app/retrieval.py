"""Semantic retrieval for RAG pipeline."""

from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import config


class Retriever:
    """Handles semantic search and document retrieval."""
    
    def __init__(self):
        """Initialize the retriever with embedding model and vector DB."""
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.client = chromadb.PersistentClient(path=str(config.vector_db_dir))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text."""
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant document chunks for a query."""
        if top_k is None:
            top_k =	config.top_k
        
        # Refresh collection reference to ensure it's up-to-date
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        query_embedding = self.query_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                retrieved_docs.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i]
                })
        
        return retrieved_docs
    
    def retrieve_with_ids(self, query: str, top_k: int = None) -> Tuple[List[str], List[Dict]]:
        """Retrieve documents and return both IDs and full doc objects."""
        docs = self.retrieve(query, top_k)
        doc_ids = [doc["metadata"]["doc_id"] for doc in docs]
        return doc_ids, docs
