CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS arxiv_embeddings_4096 (
    arxiv_code VARCHAR(20),
    doc_type VARCHAR(50),
    embedding_type VARCHAR(50),
    embedding vector(4096),
    tstp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT arxiv_embeddings_4096_pkey PRIMARY KEY (arxiv_code, doc_type, embedding_type)
);

CREATE TABLE IF NOT EXISTS arxiv_embeddings_2048 (
    arxiv_code VARCHAR(20),
    doc_type VARCHAR(50),
    embedding_type VARCHAR(50),
    embedding vector(2048),
    tstp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT arxiv_embeddings_2048_pkey PRIMARY KEY (arxiv_code, doc_type, embedding_type)
);


CREATE TABLE IF NOT EXISTS arxiv_embeddings_1024 (
    arxiv_code VARCHAR(20),
    doc_type VARCHAR(50),
    embedding_type VARCHAR(50),
    embedding vector(1024),
    tstp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT arxiv_embeddings_1024_pkey PRIMARY KEY (arxiv_code, doc_type, embedding_type)
);