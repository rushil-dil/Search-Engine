import json
import os

def prepare(index_path="index/inverted_index.json", out_dir="search_index"):
    os.makedirs(out_dir, exist_ok=True)

    postings_path = os.path.join(out_dir, "postings.jsonl")
    lexicon_path = os.path.join(out_dir, "lexicon.json")
    doc_lengths_path = os.path.join(out_dir, "doc_lengths.json")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    lexicon = {}
    doc_lengths = {}

    with open(postings_path, "w", encoding="utf-8") as out:
        for term in sorted(index):
            offset = out.tell()
            line = json.dumps(index[term], separators=(",", ":"))
            out.write(line + "\n")
            lexicon[term] = [offset, len(line.encode("utf-8")) + 1, len(index[term])]
            for doc_id, tf in index[term]:
                doc_lengths[doc_id] = doc_lengths.get(doc_id, 0) + tf

    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, separators=(",", ":"))

    with open(doc_lengths_path, "w", encoding="utf-8") as f:
        json.dump(doc_lengths, f, separators=(",", ":"))

    print("Search index prepared.")
    print(f"Terms: {len(lexicon)}")
    print(f"Documents: {len(doc_lengths)}")

if __name__ == "__main__":
    prepare()