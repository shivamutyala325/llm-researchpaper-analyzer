# embedding_index.py
import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class VectorStore:
    """
    Handles embeddings (SentenceTransformer) + FAISS index + mapping to SQLite chunk IDs.
    CPU-only version (uses faiss-cpu).
    """
    def __init__(self, db_path="papers.db", index_path="faiss.index", dim=384):
        self.db_path = db_path
        self.index_path = index_path
        self.dim = dim
        self.model = SentenceTransformer(EMBED_MODEL)

        # ensure DB connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.executescript("""
        CREATE TABLE IF NOT EXISTS faiss_mapping (
            faiss_idx INTEGER PRIMARY KEY,
            chunk_id INTEGER
        );
        """)
        self.conn.commit()

        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # use cosine similarity: normalize vectors + IndexFlatIP
            self.index = faiss.IndexFlatIP(self.dim)
            faiss.write_index(self.index, self.index_path)

    def add_embeddings(self, texts, chunk_ids, batch_size=64):
        """
        texts: list of chunk texts
        chunk_ids: list of chunk IDs (must match order of texts)
        """
        assert len(texts) == len(chunk_ids)
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(emb)  # normalize for cosine similarity
            all_embs.append(emb)
        all_embs = np.vstack(all_embs).astype("float32")

        start_pos = self.index.ntotal
        self.index.add(all_embs)

        # map FAISS positions to chunk_ids
        for i, cid in enumerate(chunk_ids):
            pos = start_pos + i
            self.cur.execute(
                "INSERT INTO faiss_mapping (faiss_idx, chunk_id) VALUES (?, ?)",
                (int(pos), int(cid))
            )
        self.conn.commit()
        faiss.write_index(self.index, self.index_path)

    def search(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb.astype("float32"), top_k)
        return D[0], I[0]

    def map_index_to_chunk_ids(self, indices):
        """Map FAISS positions back to chunk IDs from DB."""
        chunk_ids = []
        for idx in indices:
            if idx < 0:
                chunk_ids.append(None)
                continue
            self.cur.execute("SELECT chunk_id FROM faiss_mapping WHERE faiss_idx=?", (int(idx),))
            r = self.cur.fetchone()
            chunk_ids.append(r[0] if r else None)
        return chunk_ids

    def close(self):
        self.conn.close()
