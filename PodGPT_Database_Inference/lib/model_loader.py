# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BitsAndBytesConfig,
)


class Encoder:
    """
    Load embedding model and tokenizer from Hugging Face
    :param embed_model: the name of the embedding model
    :param device: the device, i.e., "cuda", "mps" or "cpu"
    :return model: the embedding model
    """
    def __init__(self, embed_model, device):
        self.embed_model = embed_model
        self.device = device

        # Load model
        if 'cuda' in device:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,  # Typically disabled for INT4
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.embed_model,
                # device_map="auto",
                # By default, cuda:0 will be used, if you don't set device_map="auto"!
                device_map=self.device,
                trust_remote_code=True,
                quantization_config=quantization_config
            )
        elif device == 'cpu':
            self.model = AutoModel.from_pretrained(self.embed_model, trust_remote_code=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model, trust_remote_code=True)


class DenseEncoder(Encoder):
    def __init__(self, embed_model, device):
        Encoder.__init__(self, embed_model, device)

        # Each query must come with a one-sentence instruction that describes the task
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def prompt_format(self, task_description, query):
        return f'Instruct: {task_description}\nQuery: {query}'

    def encode(self, texts):
        """
        Encode a list of texts into embeddings using the loaded model and tokenizer.

        :param texts: List of texts to encode
        :return: Tensor of embeddings
        """
        queries = [self.prompt_format(task_description=self.task, query=query) for query in texts]
        encodings = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024
        ).to(self.device)
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                attention_mask=attention_mask
            )
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def score_relevance(self, original_text, docs):
        queries = []
        # Now the dimension is [1]
        queries.append(self.prompt_format(task_description=self.task, query=original_text))
        for doc in docs:
            queries.append(doc['text'])  # Now the dimension is [1 + num_dense_doc + num_sparse_doc]

        encodings = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024
        ).to(self.device)
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                attention_mask=attention_mask
            )
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # embeddings[0]: the first item in the embeddings, which represents the query
            # embeddings[1:]: the relevant docs of the query (embeddings[0])
            # We use Cosine Similarity here
            # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/blob/main/README.md#transformers
            scores = embeddings[0] @ embeddings[1:].T

        # Add the score to each document
        for i, score in enumerate(scores):
            docs[i]['score'] = score.item()

        return docs


class SparseEncoder:
    def __init__(self, embed_model, device):
        self.embed_model = embed_model
        self.device = device

        # Load the model and tokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(self.embed_model, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model, trust_remote_code=True)

    def encode(self, texts):
        """
        Encode a list of texts into embeddings using the loaded model and tokenizer.

        :param texts: List of texts to encode
        :return: Tensor of embeddings
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            # For the "opensearch-project/opensearch-neural-sparse-encoding-v2-distill" model
            # "max_position_embeddings": 512
            # Reference:
            # https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini/
            # blob/main/config.json#L14
            max_length=512
        ).to(self.device)
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        special_token_ids = [self.tokenizer.vocab[token] for token in self.tokenizer.special_tokens_map.values()]
        with torch.no_grad():
            model_output = self.model(
                input_ids,
                output_hidden_states=True,
                attention_mask=attention_mask
            )
            embeddings, _ = torch.max(model_output[0] * attention_mask.unsqueeze(-1), dim=1)
            embeddings = torch.log(1 + torch.relu(embeddings))
            embeddings[:, special_token_ids] = 0

        return embeddings
