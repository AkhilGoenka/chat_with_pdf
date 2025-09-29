#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# retrieval.py
def retrieve_top_k(collection, query_embedding, top_k=2):
    """
    Retrieve top k documents from Chroma for a query embedding.
    Returns documents, ids, and metadatas.
    """
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return docs, ids, metas

