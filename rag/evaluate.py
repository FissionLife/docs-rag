"""Evaluation harness.

Six things can be measured, because a RAG system can fail at each
independently. The first five run every time; faithfulness is opt-in because
it costs one extra generation call per question.

  retrieval recall  -- did the right document reach the context at all?
                       If this fails, nothing downstream can save the answer.
  retrieval rank     -- MRR and nDCG@k: recall is binary (in the top-k or
                       not); these two reward the right document landing
                       *near the top* over merely scraping into slot 6.
                       Computed with binary relevance (one correct document
                       per question) -- graded nDCG needs graded relevance
                       judgments this eval set does not carry.
  context precision -- of the chunks retrieved, what fraction did the
                       generator actually cite? Low precision means the top-k
                       is full of plausible-looking noise around the one
                       chunk that mattered. Free to compute: citations and
                       hit counts are already produced by the steps above.
  answer accuracy   -- given good context, did the generator state the facts?
                       Checked by requiring specific strings (numbers, names)
                       rather than by fuzzy similarity, which hides errors.
  faithfulness      -- (opt-in, --faithfulness) does every cited claim
                       actually follow from the block it cites? Citation
                       verification (generate.py) checks a citation points at
                       a real block; it does not check the block supports the
                       claim. This is that check -- one Gemini judge call per
                       answer, using the client already in this project. Built
                       instead of adopting the `ragas` package: verified by
                       installing it that ragas 0.4.3 fails to import out of
                       the box (an internal import of a langchain_community
                       class that no longer exists there), and even fixed
                       pulls 100+ packages of LangChain+OpenAI for a metric
                       this is one function.
  grounding         -- did it refuse when the corpus has no answer?
                       Split into out-of-domain (easy, the gate should catch
                       it) and adjacent-absent (hard: the topic is in the
                       corpus but the specific fact is not, so the gate passes
                       and only the prompt and citation check stand in the way).

Retrieval-rank metrics are computed from a *live* retriever.search() call, not
from the cached generation answer -- they need the rank-ordered document list,
and re-running retrieval costs one embedding call (plentiful: 1000/day) rather
than one generation call (scarce: ~20/day on some models). Cached answers are
still used for accuracy and refusal, so re-adding these metrics never spends
a generation call you already paid for.
"""
import hashlib
import json
import math
import re
import sys
import time

from .backoff import DailyQuotaExceeded
from .config import EMBED_DIM, EMBED_MODEL, GEN_MODEL, ROOT, TOP_K
from .generate import judge_faithfulness
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
        "note": r.get("note"),
    }
    cache[k] = slim
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return slim


def _faithfulness_key(q: str, answer_text: str) -> str:
    """Keyed on the answer text too, so a changed answer re-judges rather
    than silently keeping a verdict for a claim that no longer exists."""
    sig = f"faith|{q}|{answer_text}|{GEN_MODEL}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


def _judge(rag: Rag, item: dict, r: dict, hits: list[dict],
          cache: dict) -> dict | None:
    """Faithfulness score for one answer, cached. None if not judgeable."""
    if r["refused"] or not r["citations"]:
        return None      # nothing was claimed against the context to judge
    k = _faithfulness_key(item["q"], r["answer"])
    if k in cache:
        return cache[k]
    verdict = judge_faithfulness(item["q"], r["answer"], hits)
    cache[k] = verdict
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return verdict


def _doc_rank(hits: list[dict], want: str | None) -> int | None:
    """1-indexed rank of `want`'s first appearance in retrieval order.

    Ranks *documents*, not chunks: several chunks of the top-k can belong to
    one document, so hits are collapsed to their first-seen doc_id before
    searching. Returns None if `want` never appears, or if there is no ground
    truth to rank against (`want is None`, a corpus-level question).
    """
    if want is None:
        return None
    seen: list[str] = []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.append(h["doc_id"])
    for rank, d in enumerate(seen, 1):
        if d == want or d.startswith(want + "-"):
            return rank
    return None


def main() -> None:
    fresh = "--fresh" in sys.argv
    want_faithfulness = "--faithfulness" in sys.argv
    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    cache = _load_cache(fresh)
    rag = Rag()
    if cache:
        print(f"(reusing {len(cache)} cached answers; --fresh to ignore)")
    if want_faithfulness:
        print("(--faithfulness: one extra judge call per answered question)")
    print(f"index: {rag.n} chunks | embed {EMBED_MODEL}@{EMBED_DIM}d | "
          f"gen {GEN_MODEL} | top_k {TOP_K}\n")

    t0 = time.time()
    recall = correct = 0
    false_refusals = []
    fact_misses = []
    reciprocal_ranks = []
    ndcg_scores = []
    precisions = []
    faithfulness_scores = []
    unfaithful = []

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

        # Document ids carry a short hash of their source, so the question
        # file names the readable prefix and we match on that. Rank comes
        # from a fresh retrieval call (embedding only, cheap) rather than the
        # cached generation answer, which only ever stored an unordered set.
        want = item["doc"]
        # For a corpus-level question (doc=None) the real context was the
        # overview manifest, not a similarity search -- use that here too,
        # so faithfulness judging (if requested) sees what was actually shown
        # to the generator rather than an empty, misleadingly-failing context.
        hits = rag.retriever.search(item["q"]) if want is not None \
            else rag.overview()
        rank = _doc_rank(hits, want)
        hit = want is None or rank is not None
        recall += hit
        if want is not None:
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            ndcg_scores.append(1.0 / math.log2(rank + 1) if rank else 0.0)
            if hits:
                precisions.append(len(r["citations"]) / len(hits))

        missing = [p for p in item["must"]
                   if not re.search(p, r["answer"], re.I)]
        ok = not r["refused"] and not missing
        correct += ok
        if r["refused"]:
            false_refusals.append(item["q"])
        elif missing:
            fact_misses.append((item["q"], missing))

        verdict = None
        if want_faithfulness:
            try:
                verdict = _judge(rag, item, r, hits, cache)
            except DailyQuotaExceeded as e:
                print(f"\n{e}")
                print("\nPartial results are cached; rerun to continue.")
                raise SystemExit(1)
            if verdict and verdict["score"] is not None:
                faithfulness_scores.append(verdict["score"])
                if verdict["score"] < 80:
                    unfaithful.append((item["q"], verdict))

        mark = "PASS" if ok else "FAIL"
        rank_str = f" rank={rank}" if want is not None else ""
        faith_str = (f" faith={verdict['score']}"
                    if verdict and verdict["score"] is not None else "")
        print(f"\n{i:>2}. [{mark}] retrieval={'hit' if hit else 'MISS'}"
              f"{rank_str} cos={r['best_cosine']} cites={r['citations']}"
              f"{faith_str}")
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
    if reciprocal_ranks:
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        ndcg = sum(ndcg_scores) / len(ndcg_scores)
        print(f"  MRR                  {mrr:6.3f}        "
              f"(1.0 = correct doc always ranked #1)")
        print(f"  nDCG@{TOP_K}               {ndcg:6.3f}        "
              f"(binary relevance -- rewards rank, not just presence)")
    if precisions:
        prec = sum(precisions) / len(precisions)
        print(f"  context precision    {prec:6.1%}        "
              f"(share of retrieved chunks actually cited)")
    print(f"  answer accuracy      {correct}/{na}   {correct / na:6.1%}")
    if want_faithfulness:
        if faithfulness_scores:
            avg = sum(faithfulness_scores) / len(faithfulness_scores)
            print(f"  faithfulness         {avg:6.1f}/100     "
                  f"({len(faithfulness_scores)} answers judged)")
        else:
            print("  faithfulness         n/a (nothing answered to judge)")
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
    if unfaithful:
        print("\n  FAITHFULNESS below 80 (citation present, claim doubtful):")
        for q, v in unfaithful:
            print(f"    - {q}  [{v['score']}/100]\n        {v['reason']}")

    # A question set is written against one specific corpus. Swap the corpus
    # and the questions quietly stop describing it: retrieval recall collapses
    # and the numbers become meaningless rather than merely bad. Say so, rather
    # than letting a low score read as a broken pipeline.
    if recall / na < 0.7 or correct / na < 0.7:
        print("\n" + "!" * 74)
        print("  Scores are low. Before debugging the pipeline, check that")
        print("  eval/questions.json still describes the documents you have")
        print("  indexed. A question set written for a different corpus scores")
        print("  near zero on retrieval however well the system works.")
        print()
        print("     uv run list             what is actually indexed")
        print('     uv run inspect "..."    what retrieval returns')
        print()
        print("  If the corpus has changed, rewrite the questions to match it.")
        print("!" * 74)


if __name__ == "__main__":
    main()
