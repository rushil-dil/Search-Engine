import json
import math
import re
import time
from nltk.stem import PorterStemmer

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STEMMER = PorterStemmer()

# BM25 parameters
K1 = 1.5
B = 0.75

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "what", "which", "who",
    "whom", "when", "where", "why", "how", "all", "any", "some", "not",
    "no", "so", "if", "as", "up", "out", "about", "than", "then", "more",
    "also", "just", "into", "such", "very", "i", "we", "you", "he", "she",
    "me", "us", "him", "her", "my", "our", "your", "his", "its"
}


def tokenize_query(query):
    tokens = []
    for raw in TOKEN_RE.findall(query.lower()):
        if raw not in STOP_WORDS:
            tokens.append(STEMMER.stem(raw))
    # fall back to full token list if stop words removed everything
    if not tokens:
        tokens = [STEMMER.stem(t.lower()) for t in TOKEN_RE.findall(query)]
    return tokens


def load_postings(term, lexicon, postings_file):
    if term not in lexicon:
        return [], 0
    offset, _, df = lexicon[term]
    postings_file.seek(offset)
    return json.loads(postings_file.readline()), df


def bm25_score(tf, df, dl, avgdl, N):
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
    return idf * tf_norm


def search(query, lexicon, doc_ids, doc_lengths, avgdl, postings_path, top_k=10):
    terms = tokenize_query(query)
    if not terms:
        return []

    N = len(doc_ids)
    postings_data = []

    with open(postings_path, "r", encoding="utf-8") as pf:
        for term in terms:
            postings, df = load_postings(term, lexicon, pf)
            if postings:
                postings_data.append((term, postings, df))

    if not postings_data:
        return []

    # Soft AND: require docs to contain at least min_match distinct query terms.
    # For 1-2 term queries require all; for longer queries allow one miss.
    min_match = len(postings_data) if len(postings_data) <= 2 else len(postings_data) - 1

    # Count how many query terms each doc has, accumulate BM25 score
    doc_term_count = {}
    scores = {}
    for term, postings, df in postings_data:
        for doc_id, tf in postings:
            dl = doc_lengths.get(str(doc_id), avgdl)
            s = bm25_score(tf, df, dl, avgdl, N)
            scores[doc_id] = scores.get(doc_id, 0) + s
            doc_term_count[doc_id] = doc_term_count.get(doc_id, 0) + 1

    # Filter to docs meeting the min_match threshold
    matching = {d for d, cnt in doc_term_count.items() if cnt >= min_match}

    # If soft AND still yields nothing, open it up to any single term
    if not matching:
        matching = set(doc_term_count.keys())

    ranked = sorted(
        ((d, scores[d]) for d in matching),
        key=lambda x: x[1],
        reverse=True
    )

    return [(doc_ids[str(d)], s) for d, s in ranked[:top_k]]


def main():
    with open("search_index/lexicon.json", "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    with open("index/doc_ids.json", "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    with open("search_index/doc_lengths.json", "r", encoding="utf-8") as f:
        doc_lengths = json.load(f)

    avgdl = sum(doc_lengths.values()) / len(doc_lengths)
    postings_path = "search_index/postings.jsonl"

    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        start = time.perf_counter()
        results = search(query, lexicon, doc_ids, doc_lengths, avgdl, postings_path)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nQuery time: {elapsed:.2f} ms")
        if not results:
            print("No results found.")
            continue
        for i, (url, score) in enumerate(results, start=1):
            print(f"{i}. {url}  score={score:.4f}")


if __name__ == "__main__":
    main()
