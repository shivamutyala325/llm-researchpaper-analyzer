# init_db.py
"""
Initialize the SQLite DB and an empty FAISS index. Run once before first use (or run app and it will auto-init).
"""
from db_helpers import PaperDB

def init_all(db_path="papers.db", index_path="faiss.index"):
    print("Initializing DB and vector store...")
    pdb = PaperDB(db_path=db_path, index_path=index_path, embed_dim=384, use_gpu_faiss=False)
    pdb.close()
    print("Initialized.")

if __name__ == "__main__":
    init_all()

