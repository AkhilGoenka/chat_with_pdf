#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# qa.py
from modules.retrieval import retrieve_top_k
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def query_with_rag(query, collection, embedder, top_k=2, chat_history=None):
    """
    Embed query, retrieve top_k chunks, build prompt with context + chat history,
    call local LLM, and return answer + sources.
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    docs, ids, metas = retrieve_top_k(collection, q_emb, top_k=top_k)

    sources = []
    for doc_id, meta, doc in zip(ids, metas, docs):
        sources.append({"id": doc_id, "page": meta.get("page"), "doc": doc})

    # build context text
    context_text = "\n\n---\n\n".join([f"(page {s['page']}): {d}" for s, d in zip(sources, docs)])

    # include last few turns from chat history
    history_text = ""
    if chat_history:
        for role, text in chat_history[-6:]:
            history_text += f"{role}: {text}\n"

    prompt = f"Context:\n{context_text}\n\nHistory:\n{history_text}\n\nQuestion: {query}\n"

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    resp = qa_model("Answer concisely. Do not repeat sentences. If you do not have any extra information to share, simply state 'I do not know'.\n" + prompt, max_new_tokens=128)
    answer = resp[0]['generated_text']

    return answer, sources

