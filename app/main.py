"""Streamlit application for RAG pipeline."""

import streamlit as st
from pathlib import Path
from app import DocumentIngestor, Retriever, Generator, config


st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Advanced RAG Pipeline")
st.markdown("An evaluated retrieval-augmented generation system with semantic search and grounded responses.")


# Initialize session state
if "ingestor" not in st.session_state:
    st.session_state.ingestor = DocumentIngestor()
if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()
if "generator" not in st.session_state:
    st.session_state.generator = Generator()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Ingestion")
    uploaded_file = st.file_uploader(
        "Upload document (PDF or JSON)",
        type=["pdf", "json"],
        help="Upload a PDF or JSON file. For JSON, include 'text' or 'content' field."
    )
    
    if uploaded_file:
        doc_id = st.text_input("Document ID", value=uploaded_file.name)
        if st.button("Ingest Document"):
            with st.spinner("Ingesting document..."):
                # Save uploaded file
                file_path = config.data_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                st.session_state.ingestor.ingest_file(file_path, doc_id)
                st.success(f"Document '{doc_id}' ingested successfully!")
    
    st.divider()
    
    st.subheader("Vector DB")
    col_count = st.session_state.ingestor.collection.count()
    st.metric("Total Chunks", col_count)
    
    if st.button("Clear Collection"):
        st.session_state.ingestor.clear_collection()
        st.success("Collection cleared!")
        st.rerun()
    
    st.divider()
    
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
    config.top_k = top_k


# Main chat interface
st.header("💬 Chat with Your Documents")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.caption(f"Sources: {', '.join(message['sources'])}")


# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant documents and generating answer..."):
            # Retrieve
            doc_ids, retrieved_docs = st.session_state.retriever.retrieve_with_ids(prompt, top_k)
            
            # Generate
            response = st.session_state.generator.generate_with_sources(prompt, retrieved_docs)
            
            # Display answer
            st.markdown(response["answer"])
            
            # Display sources
            if response["sources"]:
                st.caption(f"📖 Sources: {', '.join(response['sources'])}")
                
                # Show retrieved context
                with st.expander("View Retrieved Context"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}** (similarity: {doc['similarity']:.3f})")
                        st.text(doc["text"])
                        st.caption(f"Document ID: {doc['metadata']['doc_id']}")
                        st.divider()
            else:
                st.caption("No sources found")
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"]
            })


# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
