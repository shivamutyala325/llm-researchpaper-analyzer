# app.py
import streamlit as st
import os
from datetime import datetime

# Local imports
from utils.pdf_utils import extract_text_and_metadata
from utils.chunking import chunk_text
from db_helpers import PaperDB
from llm_utils import LLM

# ----------------------------
# Config
# ----------------------------
DB_PATH = "papers.db"
INDEX_PATH = "faiss.index"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

llm_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="LLM Research Paper Analyzer", layout="wide")
st.title("LLM Research Paper Analyzer")


# ----------------------------
# Cached initialization
# ----------------------------
@st.cache_resource
def get_db_and_llm():
    pdb = PaperDB(db_path=DB_PATH, index_path=INDEX_PATH, embed_dim=384)
    llm = LLM(device=llm_device)
    return pdb, llm


pdb, llm = get_db_and_llm()

# ----------------------------
# Sidebar menu
# ----------------------------
menu = st.sidebar.selectbox("Menu", ["Upload & Process", "Search", "Stored Papers"])

# ----------------------------
# Upload & Process Tab
# ----------------------------
if menu == "Upload & Process":
    st.header("Upload a research paper (PDF)")

    # Use a unique key for the file uploader to handle reruns
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="file_uploader")

    if uploaded_file:
        # Check if the uploaded file has changed and clear session state
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.paper_info = None
            st.session_state.save_path = None

        if st.session_state.paper_info is None:
            if st.button("Extract & Process"):
                with st.spinner("Extracting text from PDF..."):
                    save_path = os.path.join(
                        UPLOAD_DIR,
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
                    )
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.paper_info = extract_text_and_metadata(save_path)
                    st.session_state.save_path = save_path
                st.success(f"File saved and extracted: {save_path}")
                st.rerun()

    if 'paper_info' in st.session_state and st.session_state.paper_info is not None:
        meta = st.session_state.paper_info

        # Display extracted info and get user input
        st.subheader("Extracted Information")
        title = st.text_input("Title", meta.get("title", ""), key="title_input")
        authors = st.text_input("Authors", "", key="authors_input")
        abstract = st.text_area("Abstract", value=meta.get("abstract", ""), key="abstract_input")
        full_text = meta.get("full_text", "")

        # Use st.form for a better user experience and to handle state properly
        with st.form("pipeline_form"):
            st.write("Click below to start the full analysis pipeline.")
            submitted = st.form_submit_button("Start full pipeline")
            if submitted:
                # Insert paper (with duplicate detection)
                paper_id, exists = pdb.insert_paper(title, authors, abstract, st.session_state.save_path, full_text)

                if exists:
                    st.warning("⚠️ This paper has already been analyzed earlier.")
                    paper = pdb.get_paper(paper_id)
                    st.subheader("Final Summary (from DB)")
                    st.write(paper[5])  # summary
                    st.subheader("Research Gaps / Future Work (from DB)")
                    st.write(paper[7])  # gaps

                else:
                    st.info(f"Inserted paper id = {paper_id}")

                    # Chunking
                    with st.spinner("Chunking text..."):
                        chunks = chunk_text(full_text, max_words=400, overlap_words=60)
                    st.write(f"Generated {len(chunks)} chunks")

                    # Insert chunks into DB
                    with st.spinner("Inserting chunks into database..."):
                        chunk_ids = pdb.insert_chunks(paper_id, chunks)

                    # Embeddings
                    with st.spinner("Indexing embeddings..."):
                        pdb.add_chunks_embeddings(chunks, chunk_ids)
                    st.success("Embeddings indexed")

                    # Summarization
                    with st.spinner("Generating summaries..."):
                        final_summary, combined_chunk_summaries = llm.summarize_chunks_pipeline(
                            chunks, chunk_max_length=100, final_max_length=256
                        )

                    # Gap analysis
                    with st.spinner("Analyzing research gaps..."):
                        gaps = llm.research_gap_analysis(
                            final_summary + "\n\n" + combined_chunk_summaries
                        )

                    # Save results
                    pdb.save_summary_and_gaps(paper_id, final_summary, combined_chunk_summaries, gaps)
                    st.success("Paper analyzed successfully ✅")

                    st.subheader("Final Summary")
                    st.write(final_summary)
                    st.subheader("Research Gaps / Future Work")
                    st.write(gaps)

                # Clear session state to allow a new upload
                st.session_state.paper_info = None
                st.rerun()

# ----------------------------
# Search Tab
# ----------------------------
elif menu == "Search":
    st.header("Semantic Search (FAISS Embeddings)")
    query = st.text_input("Enter your search query")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

    if st.button("Search"):
        if query and pdb.vs.index.ntotal > 0:
            with st.spinner("Searching..."):
                results = pdb.semantic_search(query, top_k=top_k)
            if not results:
                st.warning("No results found. The index might be empty or the query is not relevant.")
            else:
                for r in results:
                    st.write(
                        f"Score: {r['score']:.3f} — Paper ID: {r['paper_id']} — Chunk ID: {r['chunk_id']}"
                    )
                    st.write(r["chunk_text"][:1000])
                    st.markdown("---")
        else:
            st.warning("Please enter a search query and ensure papers have been processed.")


# ----------------------------
# Stored Papers Tab
# ----------------------------
elif menu == "Stored Papers":
    st.header("Stored Papers in Database")
    rows = pdb.list_papers()

    if not rows:
        st.info("No papers processed yet.")
    else:
        for row in rows:
            pid, title, authors, abstract, summary, created_at = row
            with st.expander(f"{pid} — {title}"):
                st.write("Authors:", authors)
                st.write("Abstract:", abstract)
                st.write("Summary:", summary)
                st.caption(f"Uploaded on {created_at}")

                if st.button(f"View chunks for {pid}"):
                    chunks = pdb.get_chunks_for_paper(pid)
                    for cid, order, txt in chunks:
                        st.write(f"Chunk {order} (id={cid}):")
                        st.write(txt[:800])
