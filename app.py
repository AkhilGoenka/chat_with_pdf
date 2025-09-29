#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from modules.pdf_utils_and_chunking import extract_text_from_pdf, chunk_text
from modules.embeddings import get_model, embed_texts
from modules.vector_store import build_or_load_chroma
from modules.qa import query_with_rag
from transformers import pipeline
import os

# Page config
st.set_page_config(page_title="Chat with PDF", layout="wide")

STORAGE_DIR = "./storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Sidebar: Upload & settings
st.sidebar.title("Chat with PDF — Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=400, max_value=5000, value=800, step=100)
overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=100, step=50)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text, meta)
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "embedder" not in st.session_state:
    st.session_state.embedder = get_model()

embedder = st.session_state.embedder

# Left column: conversation
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Conversation")
    # iterate in reverse, 2 messages at a time
    history = st.session_state.history
    for i in range(len(history) - 2, -1, -2):  # start from last user message
        user_role, user_txt, _ = history[i]
        assistant_role, assistant_txt, _ = history[i + 1]

        st.markdown(f"**You:** {user_txt}")
        st.markdown(f"**Assistant:** {assistant_txt}")
        st.markdown(f"---")

# Right column: chat
with col2:
    st.header("Chat with PDF — Ask questions")

    if uploaded_file is not None:
        # Process PDF only if new
        if "collection" not in st.session_state or st.session_state.get("last_uploaded_name") != uploaded_file.name:
            with st.spinner("Extracting text and building vector store..."):
                pages = extract_text_from_pdf(uploaded_file)
                chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
                embeddings = embed_texts(embedder, [c["text"] for c in chunks])
                collection = build_or_load_chroma(chunks, embeddings, persist_directory=STORAGE_DIR)
                st.session_state["collection"] = collection
                st.session_state["last_uploaded_name"] = uploaded_file.name
                st.success("PDF ready! Ask questions below")

        # Text input for query (stored in session_state to persist)
        st.session_state.current_query = st.text_input("Ask a question about the document:", st.session_state.current_query, key="query_input")

        if st.button("Send", key="send_btn") and st.session_state.current_query:
            query = st.session_state.current_query
            with st.spinner("Retrieving and answering — this may take a few seconds"):
                answer, sources = query_with_rag(
                    query,
                    st.session_state["collection"],
                    embedder,
                    top_k=2,
                    chat_history=[(r, t) for (r, t, m) in st.session_state.history if r in ("user", "assistant")],
                )
                st.markdown(f"**Assistant:** {answer}")
                st.markdown(f"\n **Reference Text:** {sources}")
                # Append to history (meta=None to exclude sources in display)
                st.session_state.history.append(("user", query, None))
                st.session_state.history.append(("assistant", answer, None))
                # Clear current query
                st.session_state.current_query = ""

    else:
        st.info("Upload a PDF in the sidebar to get started")

    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("Clear history", key="clear_btn"):
        st.session_state.history = []

