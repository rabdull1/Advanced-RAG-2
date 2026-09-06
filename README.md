# Advanced RAG Pipeline

An evaluated retrieval-augmented generation system that retrieves relevant documents, generates grounded responses, and reports retrieval and answer-quality metrics.

## What It Does

1. **Document Ingestion**: Loads and chunks source documents from JSON files
2. **Vector Embeddings**: Generates and stores semantic embeddings using sentence transformers
3. **Semantic Retrieval**: Retrieves the most relevant document chunks using similarity search
4. **Grounded Generation**: Generates answers using LLM with retrieved context
5. **Evaluation**: Measures retrieval and response quality against a golden dataset

## Architecture

```
┌─────────────┐
│   Document  │
│   Ingestion │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunking  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Embeddings  │
│ (Sentence-  │
│ Transformers│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ChromaDB   │
│ (Vector DB) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │
│  (Semantic  │
│   Search)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Generation  │
│   (LLM)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Evaluation │
│  (Metrics)  │
└─────────────┘
```

## Demonstration

![Application demonstration](assets/demo.gif)

## Evaluation Results

| Metric | Score |
|---|---:|
| Precision@5 | ACTUAL_RESULT |
| Recall@5 | ACTUAL_RESULT |
| Mean Reciprocal Rank | ACTUAL_RESULT |
| Mean Average Precision | ACTUAL_RESULT |

**Note**: Run the evaluation script to generate actual results for your dataset.

## Quick Start

```bash
git clone https://github.com/rabdull1/advanced-rag.git
cd advanced-rag
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/main.py
```

The application will be available at `http://localhost:8501`

## Test and Evaluate

```bash
# Run tests
pytest -v

# Run evaluation
python -m evals.evaluate_retrieval
```

## Usage

### Ingest Documents

1. Prepare your documents in JSON format with a `text` or `content` field
2. Upload the file through the Streamlit sidebar
3. Provide a document ID for tracking
4. Click "Ingest Document" to process

### Query Documents

1. Type your question in the chat interface
2. The system retrieves relevant document chunks
3. An answer is generated using the retrieved context
4. View the sources and retrieved context for transparency

### Evaluate Performance

1. Create a `golden_dataset.json` in the `data/` directory
2. Format each question with expected relevant document IDs
3. Run the evaluation script to compute metrics

## Design Decisions

- **Embedding Model**: `all-MiniLM-L6-v2` provides a good balance of performance and efficiency
- **Vector Database**: ChromaDB offers persistent storage and easy integration
- **Chunking Strategy**: Overlapping chunks preserve context boundaries
- **Retrieval**: Cosine similarity for semantic matching
- **Generation**: OpenAI GPT models for high-quality responses

## Lessons Learned

- Chunk size significantly impacts retrieval quality
- Overlapping chunks help preserve context across boundaries
- Evaluation metrics are essential for iterative improvement
- Source citations build trust in generated answers

## Limitations

- Requires Gemini  API key for generation (can be replaced with local models)
- Retrieval quality depends on document quality and chunking strategy
- Faithfulness metric is simplified and may not capture all nuances
- Current implementation uses a single embedding model

## Responsible Use

This is an independent portfolio project using public or synthetic data. It contains no confidential employer, customer, patient, or operational information. The system should not be used for medical, legal, or financial decision-making without proper validation and oversight.

## Technology Stack

- **Python 3.11**: Core language
- **Streamlit**: Web UI framework
- **Sentence Transformers**: Text embeddings
- **ChromaDB**: Vector database
- **Google Gemini**: LLM generation (Gemini 1.5)
- **Pytest**: Testing framework
- **Docker**: Containerization
- **GitHub Actions**: CI/CD

## Project Structure

```
advanced-rag/
├── .github/workflows/
│   └── ci.yml              # CI/CD pipeline
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── ingestion.py        # Document ingestion and chunking
│   ├── retrieval.py        # Semantic search and retrieval
│   ├── generation.py       # LLM generation with context
│   └── main.py             # Streamlit application
├── data/
│   ├── golden_dataset.json # Golden questions for evaluation
│   └── sample_documents.json # Sample documents
├── evals/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metric functions
│   └── evaluate_retrieval.py # Evaluation script
├── tests/
│   ├── __init__.py
│   ├── test_config.py      # Configuration tests
│   ├── test_ingestion.py   # Ingestion tests
│   └── test_metrics.py     # Metrics tests
├── assets/
│   ├── architecture.png    # Architecture diagram
│   └── demo.gif            # Application demonstration
├── .env.example            # Environment variables template
├── .gitignore
├── .dockerignore
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Docker compose configuration
├── requirements.txt        # Pinned dependencies
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request.

## License

MIT License - see LICENSE file for details
