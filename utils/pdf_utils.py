# utils/pdf_utils.py
from pypdf import PdfReader
import re

def extract_text_and_metadata(pdf_path):
    """
    Extract text per page and attempt simple heuristics for title/abstract.
    Returns dict with: title, authors (empty), abstract, pages (list of {"page":int,"text":str}), full_text.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})

    full_text = "\n".join(p["text"] for p in pages)

    # Heuristic title: first non-empty line of first page longer than 10 chars
    title = ""
    if pages:
        first_page_lines = [ln.strip() for ln in pages[0]["text"].splitlines() if ln.strip()]
        for ln in first_page_lines:
            if len(ln) > 10:
                title = ln
                break

    # Heuristic abstract extraction (look for the word 'Abstract' or 'ABSTRACT')
    abstract = ""
    m = re.search(r"(?is)\babstract\b[:\s]*(.+?)(?=\n[a-zA-Z0-9]{1,50}\n|Introduction\b|1\.\s+Introduction\b|References\b|$)", full_text)
    if m:
        abstract = m.group(1).strip()
        # Trim to first 1000-2000 chars if extremely long
        abstract = abstract[:2000]

    return {
        "title": title,
        "authors": "",
        "abstract": abstract,
        "pages": pages,
        "full_text": full_text
    }

# pdf_path=r"C:\Users\shiva\Documents\books\aiml\papers_for_project\compression_transformers.pdf"
# res=extract_text_and_metadata(pdf_path)
#
# print(res['abstract'])