-- Runs once when the postgres volume is created.
-- pgvector for the RAG embeddings, in the application database.
CREATE EXTENSION IF NOT EXISTS vector;
-- Separate database for the MLflow tracking store (same server, same user).
CREATE DATABASE mlflow;
