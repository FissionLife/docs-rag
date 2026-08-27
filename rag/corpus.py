"""The corpus: documents you have added, stored as JSON on disk.

One file per document under `data/corpus/`, holding the parsed text plus where
it came from. Keeping the parsed text (rather than re-parsing on every ingest)
means a PDF is read once, and it makes the corpus inspectable -- you can open
the JSON and see exactly what the model will be allowed to read.

`loaders.py` does the parsing; this module only decides what is in the
collection.
"""
import hashlib
import json

from .config import CORPUS_DIR
from .loaders import load_source


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def add(sources: list[str], replace: bool = False) -> list[dict]:
    """Parse each source and store it. Returns the documents added."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    # The same document reaches us under several URLs -- a GitHub blob page,
    # its raw URL and the bare repo URL all resolve to one README -- and each
    # spelling hashes to a different doc_id. Without a content check they all
    # land as separate documents, and retrieval then spends several of its six
    # context slots on identical text.
    existing = {_text_hash(d["text"]): d["doc_id"] for d in documents()}
    added = []
    for src in sources:
        try:
            docs = load_source(src)
        except Exception as e:
            print(f"  FAILED  {src}\n            {e}")
            continue

        for doc in docs:
            h = _text_hash(doc["text"])
            twin = existing.get(h)
            if twin and twin != doc["doc_id"] and not replace:
                print(f"  dupe    {doc['title']}  "
                      f"(identical text already indexed as {twin})")
                continue

            path = CORPUS_DIR / f"{doc['doc_id']}.json"
            if path.exists() and not replace:
                print(f"  exists  {doc['title']}  "
                      f"(pass --replace to overwrite)")
                continue
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            existing[h] = doc["doc_id"]
            added.append(doc)
            print(f"  added   {doc['title']}  "
                  f"[{doc['kind']}, {len(doc['text']):,} chars]")
    return added


def load() -> list[dict]:
    """Every document currently in the corpus."""
    files = sorted(CORPUS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(
            "The corpus is empty. Add something first, for example:\n"
            "  uv run add https://en.wikipedia.org/wiki/Retrieval-augmented_generation\n"
            "  uv run add ./papers/report.pdf\n"
            "  uv run add ./notes/          (a whole folder)")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def documents() -> list[dict]:
    """Like load(), but returns [] instead of exiting when empty."""
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(CORPUS_DIR.glob("*.json"))]


def remove(doc_id: str) -> bool:
    path = CORPUS_DIR / f"{doc_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True
