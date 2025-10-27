from typing import List

import torch
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbedding(Embeddings):
    def __init__(self, model, device=torch.device('cpu')):
        self.model_name = model
        self.model = SentenceTransformer(self.model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:

        return self.model.encode(text)
