
def chunk_text(text, max_words=400, overlap_words=50):
    """
    Simple word-based chunker with overlap.
    Returns list of chunks (strings). Keeps order.
    """
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap_words, end) - (0 if end==len(words) else overlap_words)
        # simpler move:
        start = end - overlap_words
    return chunks

