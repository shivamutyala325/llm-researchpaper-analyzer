# llm_utils.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm

# Use flan-t5-base for a good CPU/GPU balance. If you have lots of GPU memory, you can use flan-t5-large.
SUM_MODEL = os.getenv("SUM_MODEL", "google/flan-t5-base")

class LLM:
    def __init__(self, device=None):
        """
        device: "cuda" or "cpu" or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL)
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=0)
        else:
            self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)

    def summarize_chunk(self, chunk_text, max_length=120):
        """
        Summarize a single chunk. Keep relatively short to then combine.
        """
        prompt = f"Summarize the following excerpt from an academic paper in 1-2 concise sentences:\n\n{chunk_text}"
        out = self.pipe(prompt, max_length=max_length, truncation=True)[0]["generated_text"]
        return out.strip()

    def summarize_combined(self, combined_text, max_length=256):
        """
        Produce a polished final summary from combined chunk summaries.
        """
        prompt = f"Given these chunk summaries from a research paper, produce a concise academic summary (2-4 sentences) capturing the main contributions and findings:\n\n{combined_text}"
        out = self.pipe(prompt, max_length=max_length, truncation=True)[0]["generated_text"]
        return out.strip()

    def research_gap_analysis(self, final_context_text, max_length=256):
        """
        Given the final combined summary/context, list limitations and future work as bullet points.
        """
        prompt = ("You are an expert researcher. Based on the provided summary/context, list (A) limitations or weaknesses "
                  "that are apparent and (B) actionable future work directions or unexplored areas. Output as clear bullet points.")
        prompt = prompt + "\n\n" + final_context_text
        out = self.pipe(prompt, max_length=max_length, truncation=True)[0]["generated_text"]
        return out.strip()

    def summarize_chunks_pipeline(self, chunks, chunk_max_length=120, final_max_length=256):
        """
        Summarize each chunk, join those summaries, then re-summarize to get final summary.
        Returns (final_summary, combined_chunk_summaries_text)
        """
        chunk_summaries = []
        for c in tqdm(chunks, desc="Summarizing chunks"):
            s = self.summarize_chunk(c, max_length=chunk_max_length)
            chunk_summaries.append(s)
        # join chunk summaries into one big text (with delimiters)
        combined = "\n".join(f"- {s}" for s in chunk_summaries)
        # optionally truncate combined if too long for model input
        combined_trunc = combined if len(combined) < 15000 else combined[:15000]
        final_summary = self.summarize_combined(combined_trunc, max_length=final_max_length)
        return final_summary, combined
