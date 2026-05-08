# Local RAG Demo Notes

## Project Goal

The demo project shows how a local-first RAG application can ingest documents, build a searchable index, and answer grounded questions with citations.

## Retrieval Stack

The retrieval stack combines dense FAISS search with BM25 sparse search. Dense retrieval uses local sentence-transformer embeddings, while BM25 is rebuilt from stored chunks.

## Safety Practices

API keys are not stored in SQLite. Temporary keys are held only in the Streamlit session, and runtime documents stay under local storage.

## Evaluation

Retrieval evaluation checks whether expected files or pages appear in the top retrieved chunks. The evaluation workflow does not call OpenAI.
