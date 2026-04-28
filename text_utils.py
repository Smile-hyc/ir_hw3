import html
import re
from pathlib import Path

import jieba


STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "及",
    "对",
    "中",
    "为",
    "也",
    "等",
    "一个",
    "可以",
    "通过",
}


def load_documents(folder_path):
    """Load all .txt and .md documents from a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        return []

    documents = []
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            documents.append(
                {
                    "source_file": file_path.name,
                    "text": text,
                }
            )
    return documents


def split_text(text, max_chunk_chars=260):
    """Split text into paragraph-based chunks, then split very long paragraphs by sentence."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]

    chunks = []
    for paragraph in paragraphs:
        cleaned = re.sub(r"\s+", " ", paragraph).strip()
        if not cleaned:
            continue

        if len(cleaned) <= max_chunk_chars:
            chunks.append(cleaned)
            continue

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[。！？!?])", cleaned)
            if s.strip()
        ]
        buffer = ""
        for sentence in sentences:
            if len(buffer) + len(sentence) <= max_chunk_chars:
                buffer += sentence
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sentence
        if buffer:
            chunks.append(buffer)

    return chunks


def tokenize(text):
    """Chinese tokenizer used by BM25 and TF-IDF."""
    words = []
    for token in jieba.lcut(text):
        token = token.strip()
        if not token:
            continue
        if token in STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        words.append(token)
    return words


def highlight_terms(text, terms):
    """Highlight matched query terms with HTML <mark> tags."""
    escaped_text = html.escape(text)
    clean_terms = sorted({term for term in terms if term}, key=len, reverse=True)

    for term in clean_terms:
        escaped_term = html.escape(term)
        if not escaped_term:
            continue
        pattern = re.compile(re.escape(escaped_term), re.IGNORECASE)
        escaped_text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped_text)

    return escaped_text
