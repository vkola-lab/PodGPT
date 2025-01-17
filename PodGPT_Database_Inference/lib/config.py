# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

DATABASE_CONFIG = {
    "host": "echo.bumc.bu.edu",
    "port": 3561,
    "user": "user",
    "password": "neurogpt",
    "database": "wide_sample_embedding",
    "drivername": "postgresql+psycopg2"
}

ENCODER_CONFIG = {
    # We are referring to the MTEB Leaderboard to select the best retrieval and rerank model!
    # As of December 2024, the best-performed one is the "Alibaba-NLP/gte-Qwen2-7B-instruct"
    # https://huggingface.co/spaces/mteb/leaderboard
    "dense_model": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "sparse_model": "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
    # As for the reranker model, we are using the "dense_model"
    # "reranker_model": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "rag_type": "combined",  # "sparse" "dense"
    "device": "cuda"
}

QUERY_LIMITS = {
    "dense_limit": 15,
    "sparse_limit": 15,
    "score_threshold": 0.45
}
