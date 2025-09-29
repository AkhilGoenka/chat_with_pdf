#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#vector_store.py
import chromadb

def build_or_load_chroma(chunks, embeddings, persist_directory="./storage"):
    """
    Build a Chroma vector store and add embeddings + metadata.
    Returns the collection object.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = "pdf_chunks"

    # delete existing collection for demo
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids = [c["id"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    documents = [c["text"] for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=[e.tolist() for e in embeddings]
    )

    return collection

