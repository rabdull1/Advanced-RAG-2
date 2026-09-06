"""Evaluation metrics for RAG pipeline."""

from typing import List, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate precision at K.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Precision score between 0 and 1
    """
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    
    relevant = set(relevant_ids)
    return sum(doc_id in relevant for doc_id in retrieved) / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate recall at K.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall score between 0 and 1
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate mean reciprocal rank.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        
    Returns:
        Reciprocal rank score between 0 and 1
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate average precision.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        
    Returns:
        Average precision score between 0 and 1
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            num_relevant += 1
            precisions.append(num_relevant / i)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant)


def faithfulness_score(generated_answer: str, context_docs: List[str]) -> float:
    """Calculate faithfulness score based on context support.
    
    This is a simplified faithfulness metric that checks if key terms
    in the generated answer appear in the retrieved context.
    
    Args:
        generated_answer: The generated answer text
        context_docs: List of retrieved context documents
        
    Returns:
        Faithfulness score between 0 and 1
    """
    if not context_docs:
        return 0.0
    
    # Combine all context into single text
    combined_context = " ".join(context_docs).lower()
    
    # Extract key terms from answer (simple word-level approach)
    answer_words = set(generated_answer.lower().split())
    
    # Filter out common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "must", "shall",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "through", "during", "before", "after", "above",
                  "below", "between", "under", "again", "further", "then", "once",
                  "here", "there", "when", "where", "why", "how", "all", "each",
                  "few", "more", "most", "other", "some", "such", "no", "nor",
                  "not", "only", "own", "same", "so", "than", "too", "very"}
    
    key_terms = [word for word in answer_words if word not in stop_words and len(word) > 2]
    
    if not key_terms:
        return 0.0
    
    # Check how many key terms appear in context
    supported_terms = sum(1 for term in key_terms if term in combined_context)
    
    return supported_terms / len(key_terms)
