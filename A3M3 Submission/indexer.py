import json
import os
import re
import sys
from collections import Counter

import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
WEIGHTS = {"title": 3, "h1": 3, "h2": 2, "h3": 2, "b": 2, "strong": 2}
SKIP = {"script", "style", "noscript"}
DEFAULT_WEIGHT = 1
MIN_PARTIALS = 3

STEMMER = PorterStemmer()


def weight_for_node(text_node):
    best = DEFAULT_WEIGHT
    for ancestor in text_node.parents:
        name = ancestor.name
        if not name:
            continue
        if name in SKIP:
            return 0
        w = WEIGHTS.get(name)
        if w and w > best:
            best = w
    return best


def tokenize(html):
    soup = BeautifulSoup(html, "html.parser")
    counts = Counter()
    for node in soup.find_all(string=True):
        w = weight_for_node(node)
        if w == 0:
            continue
        for raw in TOKEN_RE.findall(str(node).lower()):
            counts[STEMMER.stem(raw)] += w
    return counts


def collect_corpus(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".json"):
                files.append(os.path.join(dirpath, name))
    return files


def assign_doc_id(url_map, url):
    key = url.split("#", 1)[0]
    existing = url_map.get(key)
    if existing is not None:
        return existing
    new_id = len(url_map)
    url_map[key] = new_id
    return new_id


def add_to_buffer(buffer, doc_id, term_counts):
    for term, tf in term_counts.items():
        postings = buffer.get(term)
        if postings is None:
            buffer[term] = {doc_id: tf}
        elif doc_id in postings:
            postings[doc_id] += tf
        else:
            postings[doc_id] = tf


def flush_buffer(buffer, partials_dir, count):
    if not buffer:
        return None
    path = os.path.join(partials_dir, f"p{count:04d}.jsonl")
    with open(path, "w", encoding="utf-8") as fout:
        for term in sorted(buffer):
            postings = sorted(buffer[term].items())
            fout.write(json.dumps([term, postings], separators=(",", ":")))
            fout.write("\n")
    return path


def read_next(reader):
    line = reader.readline()
    if not line:
        return None
    return json.loads(line)


def merge_partials(partial_paths, out_path):
    readers = [open(p, encoding="utf-8") for p in partial_paths]
    cursors = [read_next(r) for r in readers]
    unique = 0
    try:
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("{")
            first = True
            while True:
                min_term = None
                for c in cursors:
                    if c is None:
                        continue
                    if min_term is None or c[0] < min_term:
                        min_term = c[0]
                if min_term is None:
                    break

                merged = {}
                for i in range(len(cursors)):
                    if cursors[i] is not None and cursors[i][0] == min_term:
                        for doc_id, tf in cursors[i][1]:
                            merged[doc_id] = merged.get(doc_id, 0) + tf
                        cursors[i] = read_next(readers[i])

                if not first:
                    out.write(",")
                first = False
                postings_out = [[d, merged[d]] for d in sorted(merged)]
                out.write(json.dumps(min_term))
                out.write(":")
                out.write(json.dumps(postings_out, separators=(",", ":")))
                unique += 1
            out.write("}")
    finally:
        for r in readers:
            r.close()
    return unique


def build_index(corpus_root, output_dir="index"):
    partials_dir = os.path.join(output_dir, "partials")
    os.makedirs(partials_dir, exist_ok=True)

    files = collect_corpus(corpus_root)
    total = len(files)
    force_every = max(1, total // MIN_PARTIALS)

    url_map = {}
    buffer = {}
    partial_paths = []
    skipped_load = 0
    skipped_parse = 0

    for n, filepath in enumerate(files, start=1):
        try:
            with open(filepath, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped_load += 1
            continue
        url = record.get("url")
        html = record.get("content")
        if not url or html is None:
            skipped_load += 1
            continue
        try:
            counts = tokenize(html)
        except Exception:
            skipped_parse += 1
            continue

        doc_id = assign_doc_id(url_map, url)
        add_to_buffer(buffer, doc_id, counts)

        if n % force_every == 0:
            path = flush_buffer(buffer, partials_dir, len(partial_paths))
            if path:
                partial_paths.append(path)
            buffer = {}

    path = flush_buffer(buffer, partials_dir, len(partial_paths))
    if path:
        partial_paths.append(path)

    out_path = os.path.join(output_dir, "inverted_index.json")
    if partial_paths:
        unique_tokens = merge_partials(partial_paths, out_path)
    else:
        unique_tokens = 0
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("{}")

    for p in partial_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(partials_dir)
    except OSError:
        pass

    doc_ids_out = {str(i): u for u, i in url_map.items()}
    with open(os.path.join(output_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids_out, f, separators=(",", ":"))

    size_kb = sum(
        os.path.getsize(os.path.join(output_dir, e))
        for e in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, e))
    ) / 1024

    indexed = len(url_map)
    processed = total - skipped_load - skipped_parse
    duplicates = processed - indexed

    #print(f"Files in corpus:       {total}")
    #print(f"Indexed documents:     {indexed}")
    #print(f"Unique tokens:         {unique_tokens}")
    #print(f"Index size on disk:    {size_kb:.2f} KB")
    #print(f"Partial files written: {len(partial_paths)}")
    #print(f"Skipped (load):        {skipped_load}")
    #print(f"Skipped (parse):       {skipped_parse}")
    #print(f"Duplicate URLs:        {duplicates}")

    return indexed, unique_tokens, size_kb


if __name__ == "__main__":
    build_index(sys.argv[1])
