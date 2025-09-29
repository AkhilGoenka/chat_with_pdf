#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pdf_utils_and_chunking.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from each page of a PDF.
    Returns a list of dicts: {"page": page_number, "text": text}
    """
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages

def chunk_text(pages, chunk_size=800, overlap=100):
    """
    Splits PDF text into chunks using RecursiveCharacterTextSplitter.
    Returns a list of dicts: {"id": page_i, "page": page_number, "text": chunk_text}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for i, c in enumerate(page_chunks):
            chunks.append({"id": f"{page['page']}_{i}", "page": page["page"], "text": c})
    return chunks

