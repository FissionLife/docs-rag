"""Grounded answering.

"Only answer from the context" is not one switch -- it is four layers, and
the model's instructions are only the weakest of them:

  1. Retrieval gate  (retrieve.py) -- if nothing clears MIN_COSINE we refuse
     without ever calling the generator. No context, no hallucination.
  2. Prompt contract -- a system instruction that forbids outside knowledge
     and demands a [n] citation on every factual sentence, plus an exact
     refusal sentinel to emit when the context falls short.
  3. Decoding -- temperature 0, so the model takes the most context-supported
     continuation rather than a fluent invented one.
  4. Post-hoc verification (this file) -- citations are parsed and checked
     against the context actually supplied. An answer that cites nothing, or
     cites a block that was never shown, is rejected and converted into a
     refusal. This is the layer that holds when the prompt is ignored.
"""
import re

# from google import genai
from google.genai import types

from .backoff import RateLimiter, with_retry
from .config import GEN_MODEL, GEN_PER_MIN
from .embed import client
from .keys import call_with_rotation

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"

REFUSAL_MESSAGE = (
    "I can't answer that from the indexed documents. "
    "The retrieved passages don't contain the information."
)

SYSTEM = f"""You answer questions strictly and only from the numbered CONTEXT \
blocks supplied in the user message.

Rules, in priority order:

1. Use the CONTEXT as your only source of facts. Your own knowledge of the \
world is not admissible, even when you are certain it is correct and even \
when the context is obviously incomplete.
2. Cite after every factual sentence, using the block numbers you drew it \
from, like [2] or [1][4]. A sentence with no citation is not allowed.
3. When the CONTEXT does not discuss the subject of the question at all, \
reply with exactly this and nothing else: {REFUSAL_TOKEN}
Apply this only to a missing *subject*. If the CONTEXT covers the subject but \
not the particular angle asked for, rules 4 and 5 apply instead and you must \
answer.
4. Partial answers are correct behaviour. Answer the part the context \
supports, cite it, and state plainly which part is not covered.
5. A question may ask for a judgement the CONTEXT never makes -- "which is \
best", "should I", "is X better than Y", "what is the top option". Do not \
refuse those outright. Report what the CONTEXT does say about the subject, \
cite it, and add one line stating that the documents make no such comparison \
or recommendation. Refuse only when the *subject* is absent, not merely the \
verdict.
6. If the context contradicts itself, report both readings with their \
citations rather than picking one.
7. Do not speculate, extrapolate beyond what is written, or fill gaps with \
plausible detail. Quote or paraphrase closely.
8. Be concise. No preamble, no restating the question."""

# The overview route has a different job -- describe the collection rather
# than extract a fact -- so it gets its own contract. Grounding is identical:
# only the supplied CONTEXT, citations on every claim, same refusal sentinel.
# What changes is that summarising the CONTEXT is explicitly *answerable*,
# because the CONTEXT is the material being asked about. Keeping this separate
# means the strict factual prompt above is never loosened to accommodate it.
SYSTEM_OVERVIEW = f"""You describe a document collection, using only the numbered CONTEXT blocks supplied in the user message.

Block [1] is the collection's manifest: every document title, its section headings, and exact counts. Every later block is the opening summary section of one document.

Rules:

1. Use the CONTEXT as your only source. Your own knowledge of these subjects is not admissible, even where you are confident.
2. Cite the blocks you draw on, like [1] or [3][7]. Every claim needs one.
3. The CONTEXT fully describes what this collection contains, so requests to summarise, describe, characterise or list it ARE answerable. Answer them. Take counts and titles verbatim from the manifest [1].
4. If asked for a specific fact the CONTEXT does not state, reply with exactly this and nothing else: {REFUSAL_TOKEN}
5. If the request is vague, answer its most useful reading and say in one line what you took it to mean. Do not refuse merely because it is terse.
6. Be organised: a short paragraph on what the collection covers, then the main themes with example titles. Do not enumerate all documents unless asked."""

_CITE = re.compile(r"\[(\d+)\]")
_limiter = RateLimiter(GEN_PER_MIN)


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] SOURCE: {h['title']}"
        + (f" > {h['section']}" if h["section"] else "")
        + f"\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )


# Models disagree on how thinking is configured: 2.5 takes thinking_budget,
# 3.x takes thinking_level, and 3.6-flash rejects both. We try the configured
# form once, and on INVALID_ARGUMENT fall back to the model's own default --
# remembering the answer so the failure is paid once per process, not per call.
_THINKING = types.ThinkingConfig(thinking_level="low")


def _generate(prompt: str, system: str = SYSTEM) -> str:
    global _THINKING

    def once(thinking):
        cfg = dict(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=1024,
            # We pass no tools; disabling AFC silences a spurious SDK warning.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True),
        )
        if thinking is not None:
            cfg["thinking_config"] = thinking
        _limiter.reserve(1)
        r = client(GEN_MODEL).models.generate_content(
            model=GEN_MODEL, contents=prompt,
            config=types.GenerateContentConfig(**cfg))
        return (r.text or "").strip()

    def attempt():
        global _THINKING
        try:
            return with_retry(lambda: once(_THINKING))
        except Exception as e:
            if _THINKING is None or "INVALID_ARGUMENT" not in str(e):
                raise
            print(f"    note: {GEN_MODEL} rejected the thinking config; "
                  f"using its default for the rest of this run")
            _THINKING = None
            return with_retry(lambda: once(None))

    # call_with_rotation retries the whole attempt() -- including the
    # thinking-config fallback above -- on a fresh key if the current one's
    # daily quota is exhausted. attempt() looks up its client fresh inside
    # once(), so it picks up the rotated key automatically on retry.
    return call_with_rotation(attempt, GEN_MODEL)


# --- faithfulness judging ---------------------------------------------
#
# Layer 4 (citation verification, above) checks that every [n] points at a
# block that was actually supplied. It does NOT check that block [n] actually
# supports the sentence citing it -- a model could cite the right block for
# the wrong reason. This is the same gap RAGAS's `faithfulness` metric
# targets. Rather than adopt the ragas package (it fails to import out of the
# box on a clean install -- ragas/llms/base.py unconditionally imports
# ChatVertexAI from a langchain_community module that no longer exists there
# -- and even fixed, pulls the full LangChain+OpenAI stack for a metric that
# needs one extra LLM call), this is that one call, using the client already
# in this file. Same metric, zero new dependencies, opt-in because it doubles
# the generation cost of an eval run.
_JUDGE_SYSTEM = """You judge whether an ANSWER's claims are actually \
supported by the CONTEXT blocks it cites. You are not asked whether the \
answer is well-written or complete -- only whether each citation is honest.

Reply with exactly two lines:
SCORE: <integer 0-100, where 100 means every cited claim is fully supported \
by the block it cites, and 0 means none are>
REASON: <one sentence naming the specific claim that failed, or "all claims \
supported" if the score is 100>"""


def judge_faithfulness(question: str, answer_text: str,
                       hits: list[dict]) -> dict:
    """Score whether `answer_text`'s citations are honestly supported.

    Returns {"score": int 0-100, "reason": str}. Costs one generation call --
    call this only when you have already decided the cost is worth paying
    (evaluate.py gates it behind --faithfulness).
    """
    context = build_context(hits)
    prompt = (f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
             f"ANSWER TO JUDGE:\n{answer_text}")
    raw = _generate(prompt, system=_JUDGE_SYSTEM)

    m = re.search(r"SCORE:\s*(\d{1,3})", raw)
    score = max(0, min(100, int(m.group(1)))) if m else None
    reason = re.search(r"REASON:\s*(.+)", raw)
    return {
        "score": score,
        "reason": reason.group(1).strip() if reason else raw[:200],
    }


def answer(question: str, hits: list[dict],
           system: str = SYSTEM) -> dict:
    """Answer from hits, then verify the answer is actually cited."""
    if not hits:
        return {"answer": REFUSAL_MESSAGE, "refused": True,
                "citations": [], "sources": []}

    context = build_context(hits)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    raw = _generate(prompt, system)

    if REFUSAL_TOKEN in raw:
        return {"answer": REFUSAL_MESSAGE, "refused": True,
                "citations": [], "sources": []}

    # --- layer 4: verify citations against what we actually supplied ------
    cited = {int(n) for n in _CITE.findall(raw)}
    valid = {n for n in cited if 1 <= n <= len(hits)}
    invalid = cited - valid

    if not valid:
        # Uncited answer => unverifiable => treated as ungrounded.
        return {"answer": REFUSAL_MESSAGE, "refused": True,
                "citations": [], "sources": [],
                "note": "answer carried no citations; rejected as ungrounded"}

    text = raw
    for n in invalid:
        text = text.replace(f"[{n}]", "")  # drop references to nonexistent blocks
    if invalid:
        text = re.sub(r" +([.,;:])", r"", re.sub(r" {2,}", " ", text))

    sources = [{
        "n": n,
        "title": hits[n - 1]["title"],
        "section": hits[n - 1]["section"],
        "url": hits[n - 1]["url"],
        "chunk_id": hits[n - 1]["chunk_id"],
        "cosine": round(hits[n - 1]["cosine"], 3),
    } for n in sorted(valid)]

    out = {"answer": text.strip(), "refused": False,
           "citations": sorted(valid), "sources": sources}
    if invalid:
        out["note"] = f"dropped invalid citation(s): {sorted(invalid)}"
    return out
