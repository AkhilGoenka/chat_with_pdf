#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

def get_model(model_name="all-MiniLM-L6-v2"):
    """Load and return the embedding model."""
    return SentenceTransformer(model_name)

def embed_texts(model, texts):
    """
    Returns normalized embeddings as a list of numpy arrays.
    """
    emb = model.encode(texts, convert_to_numpy=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb

