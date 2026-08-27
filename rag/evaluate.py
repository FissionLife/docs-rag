"""Evaluation harness.

Three things are measured, because a RAG system can fail at each
independently:

  retrieval recall  -- did the right document reach the context at all?
                       If this fails, nothing downstream can save the answer.
  answer accuracy   -- given good context, did the generator state the facts?
                       Checked by requiring specific strings (numbers, names)
                       rather than by fuzzy similarity, which hides errors.
  grounding         -- did it refuse when the corpus has no answer?
                       Split into out-of-domain (easy, the gate should catch
                       it) and adjacent-absent (hard: the topic is in the
                       corpus but the specific fact is not, so the gate passes
                       and only the prompt and citation check stand in the way).
"""
import hashlib
import json
import re
import sys
import time

from .backoff import DailyQuotaExceeded
from .config import EMBED_DIM, EMBED_MODEL, GEN_MODEL, ROOT, TOP_K
from .pipeline import Rag

QUESTIONS = ROOT / "eval" / "questions.json"
CACHE = ROOT / "eval" / ".cache.json"


def _key(q: str) -> str:
    """Cache key. Any change to the models or top_k invalidates the entry."""
    sig = f"{q}|{GEN_MODEL}|{EMBED_MODEL}|{EMBED_DIM}|{TOP_K}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


def _load_cache(fresh: bool) -> dict:
    if fresh or not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ask(rag: Rag, q: str, cache: dict) -> dict:
    """Ask, or reuse a cached answer.

    The free tier allows only ~20 generations per model per day, so a run
    that dies two thirds of the way through must not throw away the calls it
    already paid for. Every answer is written to disk as soon as it arrives.
    """
    k = _key(q)
    if k in cache:
        return cache[k]

    r = rag.ask(q)
    slim = {
        "answer": r["answer"],
        "refused": r["refused"],
        "citations": r["citations"],
        "gated": r.get("gated", False),
        "best_cosine": r.get("best_cosine"),
        "docs": sorted({h["doc_id"] for h in r["hits"]}),
        "note": r.get("note"),
    }
    cache[k] = slim
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return slim


def main() -> None:
    fresh = "--fresh" in sys.argv
    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    cache = _load_cache(fresh)
    rag = Rag()
    if cache:
        print(f"(reusing {len(cache)} cached answers; --fresh to ignore)")
    print(f"index: {rag.n} chunks | embed {EMBED_MODEL}@{EMBED_DIM}d | "
          f"gen {GEN_MODEL} | top_k {TOP_K}\n")

    t0 = time.time()
    recall = correct = 0
    false_refusals = []
    fact_misses = []

    print("=" * 78)
    print("ANSWERABLE")
    print("=" * 78)
    for i, item in enumerate(spec["answerable"], 1):
        try:
            r = _ask(rag, item["q"], cache)
        except DailyQuotaExceeded as e:
            print(f"\n{e}")
            print("\nPartial results are cached; rerun to continue.")
            raise SystemExit(1)
        hit = item["doc"] is None or item["doc"] in r["docs"]
        recall += hit

        missing = [p for p in item["must"]
                   if not re.search(p, r["answer"], re.I)]
        ok = not r["refused"] and not missing
        correct += ok
        if r["refused"]:
            false_refusals.append(item["q"])
        elif missing:
            fact_misses.append((item["q"], missing))

        mark = "PASS" if ok else "FAIL"
        print(f"\n{i:>2}. [{mark}] retrieval={'hit' if hit else 'MISS'} "
              f"cos={r['best_cosine']} cites={r['citations']}")
        print(f"    Q: {item['q']}")
        print(f"    A: {r['answer'][:260]}")
        if missing:
            print(f"    missing facts: {missing}")

    print("\n" + "=" * 78)
    print("UNANSWERABLE  (correct behaviour = refuse)")
    print("=" * 78)
    refused = 0
    leaks = []
    by_kind: dict[str, list[int]] = {}
    for i, item in enumerate(spec["unanswerable"], 1):
        try:
            r = _ask(rag, item["q"], cache)
        except DailyQuotaExceeded as e:
            print(f"\n{e}")
            print("\nPartial results are cached; rerun to continue.")
            raise SystemExit(1)
        good = r["refused"]
        refused += good
        by_kind.setdefault(item["kind"], []).append(good)
        if not good:
            leaks.append((item["q"], r["answer"]))
        how = ("gate" if r["gated"] else "model" if good else "ANSWERED")
        print(f"\n{i:>2}. [{'PASS' if good else 'LEAK'}] via={how} "
              f"cos={r['best_cosine']}")
        print(f"    Q: {item['q']}")
        print(f"    A: {r['answer'][:220]}")

    # --- summary --------------------------------------------------------
    na, nu = len(spec["answerable"]), len(spec["unanswerable"])
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  retrieval recall@{TOP_K}   {recall}/{na}   "
          f"{recall / na:6.1%}")
    print(f"  answer accuracy      {correct}/{na}   {correct / na:6.1%}")
    print(f"  refusal rate         {refused}/{nu}   {refused / nu:6.1%}")
    for kind, results in sorted(by_kind.items()):
        print(f"      {kind:<18} {sum(results)}/{len(results)}")
    print(f"  false refusals       {len(false_refusals)}")
    print(f"  elapsed              {time.time() - t0:.0f}s")

    if false_refusals:
        print("\n  refused but should have answered:")
        for q in false_refusals:
            print(f"    - {q}")
    if fact_misses:
        print("\n  answered but missed required facts:")
        for q, m in fact_misses:
            print(f"    - {q}\n        missing {m}")
    if leaks:
        print("\n  GROUNDING LEAKS (answered from outside the corpus):")
        for q, a in leaks:
            print(f"    - {q}\n        {a[:160]}")


if __name__ == "__main__":
    main()
