# Chat with PDF
An interactive Streamlit app that lets you **upload a PDF, index its contents, and ask questions**.  
Answers are grounded in retrieved document chunks and include citations.  

**Objective** - Upload a PDF, index it, and chat with it. Answers should be grounded in retrieved chunks with citations.

**How to run (locally)**
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

**Technical Stack**
1. Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
2. LLM: google/flan-t5-small (local, lightweight)
3. Vector DB: ChromaDB (local persistence)
4. Text Splitting: Recursive chunking (800 chars + 100 overlap)
5. Retrieval: Simple nearest neighbour, top 2 by default

**Project Structure**
```bash
chat-with-pdf/
├── app.py                          # Main application entry point
├── requirements.txt                # Dependencies
├── modules/                        # Core business logic
│   ├── pdf_utils_and_chunking.py   # PDF processing & text splitting
│   ├── embeddings.py               # Text embedding management
│   ├── vector_store.py             # Vector database operations
│   ├── retrieval.py                # Similarity search & chunk retrieval
│   └── qa.py                       # Question-answering engine
└── storage/                        # Persistent data
```


**Chunking Strategy**
1. Uses RecursiveCharacterTextSplitter from LangChain-style logic.
2. Configurable chunk size (default: 800 characters) with overlap (default: 100 characters).
3. Ensures semantic continuity across chunks for better retrieval.

**How conversation history is kept** - Streamlit st.session_state.history stores tuples (role, text, meta) which are displayed in the left column. Chat history is also appended to the prompt (last 6 turns) so follow-up questions have context.

**Known Limitations**
1. Local LLM (flan-t5-small) is lightweight → answers may lack depth.
2. Retrieval is purely similarity-based, no advanced re-ranking.
3. Currently supports single-PDF workflow only.
4. Limited citation formatting.

**Basic Evaluation Results**

| Screenshot #  | Score  |
| ------------- | ------ |
|       1       |   4    |
|       2       |   9    |
|       3       |   10   |
|       4       |   2    |
|       5       |   9    |

**LLM assistance** - https://chatgpt.com/share/68db13ed-68d4-8009-be8e-9e967ac6259e

Appreciate your feedback!
