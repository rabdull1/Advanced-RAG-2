"""LLM generation with retrieved context for RAG pipeline."""

from typing import List, Dict, Optional
import google.generativeai as genai
from app.config import config


class Generator:
    """Handles LLM generation with context from retrieved documents."""
    
    def __init__(self):
        """Initialize the generator with Gemini client."""
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.client = genai.GenerativeModel(config.model_name)
        else:
            self.client = None
    
    def build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Build prompt with query and retrieved context."""
        context_text = "\n\n".join([
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context_text}

Question: {query}

Answer the question using only the information from the context above. If the context doesn't contain enough information to answer the question, say so clearly. Be concise and factual."""
        
        return prompt
    
    def generate(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using LLM with retrieved context."""
        if not self.client:
            return "Error: Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
        
        if not context_docs:
            return "No relevant documents found to answer the question."
        
        prompt = self.build_prompt(query, context_docs)
        
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                )
            )
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_with_sources(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate answer with source citations."""
        answer = self.generate(query, context_docs)
        sources = list(set([doc["metadata"]["doc_id"] for doc in context_docs]))
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
