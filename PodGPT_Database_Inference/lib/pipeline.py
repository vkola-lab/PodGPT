# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import time

from sqlalchemy import text, select
from pgvector.sqlalchemy import SparseVector

from lib.database import get_session, PMCArticles
from lib.model_loader import DenseEncoder, SparseEncoder
from lib.config import ENCODER_CONFIG, QUERY_LIMITS
from utils.utils import format_documents


class Pipeline:
    """
    A pipeline class that integrates dense and sparse retrieval mechanisms with reranking for
    efficient document retrieval and processing.
    """
    def __init__(self):
        """
        Initializes the pipeline with dense encoder, sparse encoder, and reranker models.
        """
        self.dense_encoder = DenseEncoder(ENCODER_CONFIG["dense_model"], ENCODER_CONFIG["device"])
        self.sparse_encoder = SparseEncoder(ENCODER_CONFIG["sparse_model"], ENCODER_CONFIG["device"])
        self.rag_type = ENCODER_CONFIG["rag_type"]

    def retrieve_documents(self, original_text):
        """
        Retrieves and processes documents based on the input text.
        Combines dense and sparse retrieval methods, followed by reranking.

        :param original_text: list, a list of user queries or input texts for document retrieval.
        :return: list, a list of formatted documents as JSON objects.
        """
        if not original_text:
            return []

        # Step 1: Perform dense retrieval
        if self.rag_type in (["dense", "combined"]):
            start_time = time.time()
            dense_docs, dense_ids = self._run_dense_retrieval(original_text)
            print(f"Dense retrieval time: {time.time() - start_time:.2f} seconds")
        else:
            dense_docs = None
            dense_ids = None

        # Step 2: Perform sparse retrieval
        if self.rag_type in (["sparse", "combined"]):
            start_time = time.time()
            sparse_docs, sparse_ids = self._run_sparse_retrieval(original_text)
            print(f"Sparse retrieval time: {time.time() - start_time:.2f} seconds")
        else:
            sparse_docs = None
            sparse_ids = None

        # Step 3: Rerank combined results from dense and sparse retrieval
        start_time = time.time()
        if self.rag_type == "combined":
            retrieved_docs = [[x + y] for x, y in zip(dense_docs, sparse_docs)]
            retrieved_ids = [[x + y] for x, y in zip(dense_ids, sparse_ids)]
        else:
            retrieved_docs = dense_docs if self.rag_type == "dense" else sparse_docs
            retrieved_ids = dense_ids if self.rag_type == "dense" else sparse_ids
        ranked_docs = self._rerank(original_text, retrieved_docs, retrieved_ids)
        print(f"Reranking time: {time.time() - start_time:.2f} seconds")

        # Step 4: Filter and format the documents for output
        top_docs = []
        for item in ranked_docs:
            top_item_docs = [doc for doc in item if doc['score'] >= QUERY_LIMITS["score_threshold"]]
            formatted_documents = format_documents(top_item_docs)
            top_docs.append(formatted_documents)

        return top_docs

    def _run_dense_retrieval(self, original_text):
        """
        Performs dense retrieval by encoding the text and calculating similarity with database embeddings.

        :param original_text: list, a list of user queries or input text for dense retrieval.
        :return: tuple, a pair of lists containing document texts and their respective IDs.
        """
        docs = []
        ids = []
        # Generate dense embeddings for the input text
        embeddings = self.dense_encoder.encode(original_text)
        for embedding in embeddings:
            with get_session() as session:
                # We will use the Cosine Similarity between embeddings
                # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/blob/main/README.md#transformers
                query = text(f"""
                        SELECT *
                        FROM pmcarticles
                        ORDER BY embedding_gte_qwen2_7b_instruct <=> ((:embedding)::halfvec)
                        LIMIT {QUERY_LIMITS["dense_limit"]};
                """)
                result = list(session.execute(query, {'embedding': embedding.tolist()}))

                # Collect document texts and IDs from the results
                doc = [row[3] for row in result]
                id = [row[2] for row in result]

                docs.append(doc)
                ids.append(id)

        return docs, ids

    def _run_sparse_retrieval(self, original_text):
        """
        Performs sparse retrieval using sparse encoding and inner product similarity.

        :param original_text: list, a list of user queries or input text for sparse retrieval.
        :return: tuple, a pair of lists containing document texts and their respective IDs.
        """
        docs = []
        ids = []

        # Generate sparse embeddings for the input text
        embeddings = self.sparse_encoder.encode(original_text)
        for embedding in embeddings:
            with get_session() as session:
                # We will use the Inner Product between embeddings
                # https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini
                # #usage-huggingface
                query = (
                    select(PMCArticles.text, PMCArticles.sparse_vector, PMCArticles.pmc_id)
                    .order_by(PMCArticles.sparse_vector.max_inner_product(SparseVector(embedding)))
                    .limit(QUERY_LIMITS["sparse_limit"])
                )
                result = list(session.execute(query))

                # Collect document texts and IDs from the results
                doc = [row[0] for row in result]
                id = [row[2] for row in result]

                docs.append(doc)
                ids.append(id)

        return docs, ids

    def _rerank(self, original_text, docs, doc_ids):
        """
        Reranks the documents using a reranker model based on relevance scores.

        :param original_text: list, a list of user queries or input text for relevance scoring.
        :param docs: list, a list of document texts to be reranked.
        :param doc_ids: list, a list of document IDs corresponding to the texts.
        :return: list, a list of ranked documents with their scores.
        """
        ranked_results = []
        for index, query in enumerate(original_text):
            # Prepare documents in a format suitable for reranking
            formatted_docs = [
                {'text': doc, 'pmc_id': doc_id} for doc, doc_id in zip(docs[index][0], doc_ids[index][0])
            ]
            if not formatted_docs:
                ranked_results.append([])
            else:
                # Score and sort documents by relevance
                ranked_result = self.dense_encoder.score_relevance(query, formatted_docs)
                ranked_result = sorted(ranked_result, key=lambda x: x['score'], reverse=True)
                ranked_results.append(ranked_result)

        return ranked_results
