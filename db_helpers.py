# db_helpers.py
import sqlite3
from embedding_index import VectorStore
import os
import hashlib

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    authors TEXT,
    abstract TEXT,
    file_path TEXT,
    file_hash TEXT UNIQUE,
    summary TEXT,
    combined_chunk_summaries TEXT,
    gaps TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER,
    chunk_text TEXT,
    chunk_order INTEGER,
    start_page INTEGER DEFAULT NULL,
    end_page INTEGER DEFAULT NULL,
    FOREIGN KEY(paper_id) REFERENCES papers(id)
);
"""

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class PaperDB:
    def __init__(self, db_path="papers.db", index_path="faiss.index", embed_dim=384):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.executescript(DB_SCHEMA)
        self.conn.commit()
        # always use FAISS-CPU
        self.vs = VectorStore(db_path=self.db_path, index_path=index_path, dim=embed_dim)

    def insert_paper(self, title, authors, abstract, file_path, full_text):
        file_hash = compute_hash(full_text)
        self.cur.execute("SELECT id FROM papers WHERE file_hash=?", (file_hash,))
        row = self.cur.fetchone()
        if row:
            return row[0], True
        self.cur.execute(
            "INSERT INTO papers (title, authors, abstract, file_path, file_hash) VALUES (?, ?, ?, ?, ?)",
            (title, authors, abstract, file_path, file_hash)
        )
        self.conn.commit()
        return self.cur.lastrowid, False

    def insert_chunks(self, paper_id, chunk_texts):
        chunk_ids = []
        for i, txt in enumerate(chunk_texts):
            self.cur.execute(
                "INSERT INTO chunks (paper_id, chunk_text, chunk_order) VALUES (?, ?, ?)",
                (paper_id, txt, i)
            )
            chunk_ids.append(self.cur.lastrowid)
        self.conn.commit()
        return chunk_ids

    def add_chunks_embeddings(self, chunk_texts, chunk_ids):
        self.vs.add_embeddings(chunk_texts, chunk_ids)

    def semantic_search(self, query, top_k=5):
        D, I = self.vs.search(query, top_k=top_k)
        chunk_ids = self.vs.map_index_to_chunk_ids(I)
        results = []
        for score, cid in zip(D, chunk_ids):
            if cid is None:
                continue
            self.cur.execute("SELECT chunk_text, paper_id FROM chunks WHERE id=?", (cid,))
            r = self.cur.fetchone()
            if r:
                results.append({
                    "score": float(score),
                    "chunk_id": cid,
                    "paper_id": r[1],
                    "chunk_text": r[0]
                })
        return results

    def save_summary_and_gaps(self, paper_id, final_summary, combined_summaries_text, gaps_text):
        self.cur.execute(
            "UPDATE papers SET summary=?, combined_chunk_summaries=?, gaps=? WHERE id=?",
            (final_summary, combined_summaries_text, gaps_text, paper_id)
        )
        self.conn.commit()

    def list_papers(self):
        self.cur.execute("SELECT id, title, authors, abstract, summary, created_at FROM papers ORDER BY created_at DESC")
        return self.cur.fetchall()

    def get_paper(self, paper_id):
        self.cur.execute(
            "SELECT id, title, authors, abstract, file_path, summary, combined_chunk_summaries, gaps FROM papers WHERE id=?",
            (paper_id,)
        )
        return self.cur.fetchone()

    def get_chunks_for_paper(self, paper_id):
        self.cur.execute(
            "SELECT id, chunk_order, chunk_text FROM chunks WHERE paper_id=? ORDER BY chunk_order ASC",
            (paper_id,)
        )
        return self.cur.fetchall()

    def close(self):
        self.vs.close()
        self.conn.close()
