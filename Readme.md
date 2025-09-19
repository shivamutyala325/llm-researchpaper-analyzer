# LLM Research Paper Analyzer

A local Streamlit application that extracts text from academic PDFs, generates chunk-level and final summaries, performs research-gap analysis, and provides semantic search over processed papers using FAISS embeddings.

## Features
- Upload PDFs (Streamlit)
- Extract text and metadata (PyPDF)
- Chunking with overlap
- Chunk-level summarization → combined final summary (Flan-T5)
- Research gap & future-work analysis
- Semantic search using sentence-transformers (`all-MiniLM-L6-v2`) + FAISS
- SQLite database to store papers, chunks, summaries, and mappings

## Project structure

llm-paper-analyzer/
├─ app.py
├─ init_db.py
├─ requirements.txt
├─ README.md
├─ embedding_index.py
├─ db_helpers.py
├─ llm_utils.py
└─ utils/
├─ pdf_utils.py
└─ chunking.py


## Setup

1. Create, activate a virtual environment and run:
```bash
python -m venv venv
# linux/mac
source venv/bin/activate
# windows
venv\Scripts\activate

pip install -r requirements.txt

python init_db.py

streamlit run app.py
