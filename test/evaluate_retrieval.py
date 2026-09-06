"""Evaluation script for RAG pipeline retrieval metrics."""

import json
from pathlib import Path
from typing import Dict, List
from app import Retriever
from evals.metrics import precision_at_k, recall_at_k, reciprocal_rank, average_precision


def load_golden_dataset(file_path: Path) -> List[Dict]:
    """Load golden dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_retrieval(retriever: Retriever, golden_dataset: List[Dict], k: int = 5) -> Dict:
    """Evaluate retrieval performance against golden dataset.
    
    Args:
        retriever: Retriever instance
        golden_dataset: List of golden questions with expected results
        k: Number of top results to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    results = {
        "precision_at_k": [],
        "recall_at_k": [],
        "mrr": [],
        "map": [],
        "per_query": []
    }
    
    for item in golden_dataset:
        question = item["question"]
        relevant_ids = item["relevant_document_ids"]
        query_id = item.get("id", "unknown")
        
        # Retrieve documents
        doc_ids, _ = retriever.retrieve_with_ids(question, top_k=k)
        
        # Calculate metrics
        precision = precision_at_k(doc_ids, relevant_ids, k)
        recall = recall_at_k(doc_ids, relevant_ids, k)
        rr = reciprocal_rank(doc_ids, relevant_ids)
        ap = average_precision(doc_ids, relevant_ids)
        
        results["precision_at_k"].append(precision)
        results["recall_at_k"].append(recall)
        results["mrr"].append(rr)
        results["map"].append(ap)
        
        results["per_query"].append({
            "id": query_id,
            "question": question,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": rr,
            "average_precision": ap,
            "retrieved_ids": doc_ids,
            "relevant_ids": relevant_ids
        })
    
    # Calculate aggregate metrics
    results["avg_precision_at_k"] = sum(results["precision_at_k"]) / len(results["precision_at_k"])
    results["avg_recall_at_k"] = sum(results["recall_at_k"]) / len(results["recall_at_k"])
    results["mean_reciprocal_rank"] = sum(results["mrr"]) / len(results["mrr"])
    results["mean_average_precision"] = sum(results["map"]) / len(results["map"])
    
    return results


def print_evaluation_results(results: Dict):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*60)
    print("RAG PIPELINE EVALUATION RESULTS")
    print("="*60)
    print(f"\nAggregate Metrics (K={results['per_query'][0]['retrieved_ids'].__len__() if results['per_query'] else 5}):")
    print("-" * 60)
    print(f"{'Metric':<30} {'Score':>15}")
    print("-" * 60)
    print(f"{'Precision@K':<30} {results['avg_precision_at_k']:>15.4f}")
    print(f"{'Recall@K':<30} {results['avg_recall_at_k']:>15.4f}")
    print(f"{'Mean Reciprocal Rank':<30} {results['mean_reciprocal_rank']:>15.4f}")
    print(f"{'Mean Average Precision':<30} {results['mean_average_precision']:>15.4f}")
    print("-" * 60)
    
    print(f"\nPer-Query Results:")
    print("-" * 60)
    for query_result in results["per_query"]:
        print(f"\nQuery ID: {query_result['id']}")
        print(f"Question: {query_result['question']}")
        print(f"  Precision@K: {query_result['precision_at_k']:.4f}")
        print(f"  Recall@K: {query_result['recall_at_k']:.4f}")
        print(f"  MRR: {query_result['mrr']:.4f}")
        print(f"  AP: {query_result['average_precision']:.4f}")
        print(f"  Retrieved: {query_result['retrieved_ids']}")
        print(f"  Relevant: {query_result['relevant_ids']}")
    
    print("\n" + "="*60 + "\n")


def save_evaluation_results(results: Dict, output_path: Path):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Initialize retriever
    retriever = Retriever()
    
    # Load golden dataset
    golden_dataset_path = Path(__file__).parent.parent / "data" / "golden_dataset.json"
    
    if not golden_dataset_path.exists():
        print(f"Golden dataset not found at {golden_dataset_path}")
        print("Please create a golden_dataset.json file in the data/ directory.")
        exit(1)
    
    golden_dataset = load_golden_dataset(golden_dataset_path)
    
    # Run evaluation
    results = evaluate_retrieval(retriever, golden_dataset, k=5)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "evaluation_results.json"
    save_evaluation_results(results, output_path)
