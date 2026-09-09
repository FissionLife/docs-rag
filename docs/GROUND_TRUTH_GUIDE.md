# Ground Truth: a practical guide

This document stands on its own — share it with anyone building or testing a
system that's supposed to get things right, RAG or otherwise. It doesn't
assume you've read anything else in this repo. Where it uses a worked
example, it's pulled from a real, working project so every claim is checkable
rather than hypothetical, but the ideas apply far beyond RAG: evaluating a
search engine, a classifier, a chatbot, a recommendation system — anything
you're going to claim "works" needs this.

**Contents**
1. [What ground truth actually is](#1-what-ground-truth-actually-is)
2. [Why it's the whole game](#2-why-its-the-whole-game)
3. [The five shapes ground truth takes](#3-the-five-shapes-ground-truth-takes)
4. [How to actually author it well](#4-how-to-actually-author-it-well)
5. [The trap nobody warns you about](#5-the-trap-nobody-warns-you-about)
6. [A worked example, fully annotated](#6-a-worked-example-fully-annotated)
7. [Matching ground truth to the metric that needs it](#7-matching-ground-truth-to-the-metric-that-needs-it)
8. [Common mistakes, and how to catch them](#8-common-mistakes-and-how-to-catch-them)
9. [Cheat sheet](#9-cheat-sheet)

---

## 1. What ground truth actually is

**Ground truth is the answer key.** It's whatever a human (or some other
authoritative, trusted source) decided was correct, written down *before* the
system under test ever sees the question. Every automated evaluation you'll
ever run — a RAG pipeline, a search ranking, a spam filter, a medical
diagnosis model — works the same way underneath: run the system, compare its
output to the answer key, count how often they agree.

Take that apart and there are exactly two things happening:

1. **A human verified something was true, in advance, by checking the actual
   source** — not by guessing, not by asking another AI, not by "it sounds
   right."
2. **A script compares the system's output to that pre-verified fact.**

The script is the easy part. Almost all of the value — and almost all of the
work — is in step 1, done carefully, by a person who actually checked.

## 2. Why it's the whole game

Without ground truth, "does this work?" has no honest answer — only vibes.
You can watch a system produce fluent, confident-sounding output all day and
learn nothing about whether it's *correct*, because fluency and correctness
are completely different properties and a broken system can have plenty of
the first with none of the second.

Ground truth converts "this looks right" into "this system got 22 out of 22
questions correct, and here specifically are the two facts it stated
correctly that a fuzzy read-through would have missed were subtly wrong."
That difference — a specific, falsifiable, defensible number instead of an
impression — is the entire reason evaluation exists as a discipline.

One more thing worth internalizing early: **a claim without ground truth
behind it is not a fact about the system, it's a hope.** "It works pretty
well" is a hope. "18/20 exact-fact-match, verified against the source
documents by hand" is a fact.

## 3. The five shapes ground truth takes

Ground truth isn't one thing — it takes different shapes depending on what
you're actually trying to verify. Knowing which shape you need is most of
the design work.

### a) Retrieval ground truth — *which* source should be found

The correct document, passage, or record that a search/retrieval step should
surface. Binary at its simplest: did the system find it, yes or no.

```json
{"q": "What discount do EC2 Spot Instances offer?", "doc": "amazon-ec2"}
```

### b) Fact-fragment ground truth — a literal string that must appear

Cheaper to author than a full reference answer, and it catches something a
"does this sound plausible" read-through won't: a *specific wrong number or
name* hiding inside an otherwise fluent, confident sentence.

```json
{"must": ["90%"]}
```

An answer that says *"Spot Instances offer a significant discount"* sounds
fine and is worthless as a verified fact. An answer that must contain the
literal string `"90%"` either states the real number or it doesn't — no room
for a fuzzy pass.

### c) Full reference-answer ground truth — the complete correct answer

A human-written correct answer in full, not just a fragment:

> "EC2 Spot Instances offer discounts of up to 90% compared to On-Demand
> pricing, in exchange for the possibility that AWS can reclaim the instance
> with short notice."

This is a materially bigger authoring job than (b) — a full, correct sentence
per question instead of a keyword — and it's what more sophisticated
automated graders (an LLM comparing the system's answer to this reference)
actually need. Don't build it until you need it; (b) already catches most
factual errors for a fraction of the effort.

### d) Refusal / verdict ground truth — "the correct behavior is *no answer*"

Not every question should be answered. Sometimes the ground truth isn't a
fact to match, it's a **verdict**: *this system should say "I don't know."*

```json
{"q": "What is the capital city of Peru?", "kind": "out-of-domain"}
```

The human work here is checking, in advance, that the source material
genuinely doesn't cover this — so that if the system answers anyway, that's
a real, provable failure (a *leak*), not a debatable judgment call.

### e) Graded relevance ground truth — not just right/wrong, but *how* right

For ranking-quality metrics (like nDCG in its full form), ground truth can
be a *score* per candidate, not a single correct answer: this document is a
5/5 match, that one's a 2/5, that one's irrelevant. Expensive to author
properly — it needs a human to grade every candidate, not just confirm one
answer — which is why simpler binary versions (present/absent) are the
common, pragmatic substitute; more on that trade-off in §7.

## 4. How to actually author it well

This is the part that determines whether your evaluation means anything.

1. **Read the actual source before writing the question.** Not "I'm pretty
   sure this is roughly right" — open the document, find the sentence, copy
   the real number or name. If you can't point to where a fact came from,
   you don't have ground truth, you have a guess wearing ground truth's
   clothes.

2. **For a "should refuse" case, verify the absence, don't assume it.**
   `grep` the source material for the term before writing an unanswerable
   question. It's surprisingly easy to *assume* something isn't covered and
   be wrong — and a false "should refuse" question silently corrupts your
   refusal-rate metric by penalizing the system for correctly answering
   something you mistakenly thought was unanswerable.

3. **Write specific, falsifiable assertions.** "Discusses pricing" is not
   verifiable — almost any answer can be argued to loosely satisfy it. "90%"
   is verifiable — it's either in the text or it isn't. The tighter the
   assertion, the more the resulting score actually means.

4. **Cover the hard case, not just the easy one.** The easy negative
   (something totally unrelated) barely tests anything — most systems
   reject it trivially. The hard negative — a question about a topic the
   source *does* cover, asking for a specific detail it *doesn't* state — is
   where a system's honesty is actually tested. If every one of your
   "should refuse" examples is an easy one, your refusal metric is
   measuring almost nothing.

5. **Keep it light enough to actually maintain.** A fact fragment (§3b)
   costs a sentence of your time. A full reference answer (§3c) costs a
   paragraph, correctly written, fact-checked, per question. Match the
   authoring cost to what the metric you're computing actually needs —
   building (c) when (b) would answer your question is wasted, unmaintained
   effort waiting to happen.

## 5. The trap nobody warns you about

**Ground truth is coupled to the exact thing it was written to test.**
Change that thing — swap the corpus, retrain the model, redeploy a different
version — and the ground truth doesn't become *wrong*, it becomes
**meaningless**, and those two failure modes produce visually identical
output: a score.

Concretely: a question set written against one document collection, when
pointed at an *entirely different* collection, doesn't fail loudly. It just
quietly scores near zero on everything, and that number reads exactly like
"the system is broken" even though the system might be working perfectly —
it's the ground truth that no longer describes what's actually being tested.

The fix has two parts:

- **A score is only meaningful stated next to what it was measured against.**
  "95% accuracy" with no mention of what dataset, what version, what date is
  a decoration, not a claim. Always record the configuration alongside the
  number.
- **When the system under test changes materially, the ground truth has to
  be revisited — not just re-run.** Not doing this is the single most common
  way a team ends up trusting a number that stopped being true months ago.

## 6. A worked example, fully annotated

This is real, from a working RAG pipeline (a question-answering system built
on 244 real documents about AWS's cloud services) — every example below is
an actual entry from its evaluation file, not simplified for the guide.

**Retrieval + fact-fragment ground truth, combined** — the common case:

```json
{
  "q": "What discount do EC2 Spot Instances offer compared to On-Demand prices?",
  "doc": "amazon-ec2",
  "must": ["90%"]
}
```

Two independent checks come from one entry: did retrieval find the `amazon-ec2`
document (§3a)? Does the final answer contain the literal fact `"90%"` (§3b)?
A system could pass one and fail the other — find the right document but
misstate the number, or state the right number by luck while never actually
retrieving the source. Both matter, and this shape catches both.

**A tighter fact-fragment**, because "close enough" would hide a real error:

```json
{
  "q": "What durability does Amazon S3 offer?",
  "doc": "amazon-simple-storage-service",
  "must": ["99.999999999"]
}
```

Eleven nines. A system that says "very high durability" sounds fine and
fails this check — correctly, because a genuinely useful answer needed the
actual number, and "very high" isn't verifiable against anything.

**A corpus-level question**, where "the right document" doesn't apply:

```json
{
  "q": "summarise this collection",
  "doc": null,
  "must": ["AWS|Amazon", "24[0-9]|service"]
}
```

`doc: null` is itself part of the ground truth here — it's an explicit
statement that this question isn't about finding one source, and the
evaluation script has to know that going in rather than guessing. The `must`
patterns are regexes (`24[0-9]` matches "240"–"249"), loose enough to allow
a correctly-varying summary while still requiring it to state something
true and specific about the collection's actual size.

**Easy refusal ground truth** — an out-of-domain question:

```json
{"q": "What is the capital city of Peru?", "kind": "out-of-domain"}
```

**Hard refusal ground truth** — the one that actually tests something,
exactly per the advice in §4.4:

```json
{"q": "What is the maximum execution timeout for an AWS Lambda function?", "kind": "adjacent-absent"}
```

This corpus genuinely discusses AWS Lambda at length — so a weak system will
find Lambda-related content, judge it "relevant enough," and either state a
plausible-but-invented timeout or, worse, recall the real number from its
own general training rather than from the supplied documents. Verifying,
before writing this question, that the actual indexed page *never states*
a timeout number was the one-line `grep` check that makes this question
worth anything at all.

## 7. Matching ground truth to the metric that needs it

Different metrics need different *shapes* of ground truth from §3 — picking
a metric without checking what it actually requires is how projects end up
either under-testing or over-building.

| Metric | Answers | Needs (from §3) |
|---|---|---|
| Recall@k | Was the right source found at all, top-k? | (a) — binary |
| MRR / nDCG (binary) | *Where* did it rank, not just whether it appeared | (a) — same binary ground truth, just read for rank instead of presence |
| nDCG (graded, full form) | How good was the whole ranking, with partial credit | (e) — graded relevance per candidate, expensive |
| Exact-fact accuracy | Are the specific facts correct? | (b) — cheap, catches wrong numbers/names directly |
| Context recall, answer correctness (frameworks like RAGAS) | Does the answer match a full correct reference? | (c) — a complete reference answer, the expensive one |
| Refusal rate | Does the system correctly decline when it should? | (d) — a verdict, not a fact |

The practical lesson: **recall, rank-aware metrics, and exact-fact accuracy
are all obtainable from the cheap ground truth shapes** — (a) and (b) — which
is why they're usually the first metrics worth building. Full
reference-answer metrics are genuinely more informative in some ways, but
they cost a paragraph of verified human writing per question, not a
sentence — budget for that difference before committing to a framework that
assumes you already have it.

## 8. Common mistakes, and how to catch them

- **Using another model's output as ground truth.** If you generate your
  "correct" answers with an AI and grade a different AI's answers against
  them, you've built a circularity, not an evaluation — you're measuring
  agreement between two models, neither of which was ever checked against
  reality. Ground truth has to trace back to a verified source, not another
  guess.
- **Letting ground truth silently drift out of date.** The source material
  changed, the correct answer changed with it, nobody re-verified the eval
  question. Catch it by re-reading a sample of your ground truth periodically
  against the current source, not just trusting it forever once written.
- **Writing ambiguous questions with more than one defensible answer.** If
  two careful, honest people would disagree on what the "correct" answer is,
  the question isn't ready — fix the question or narrow the assertion, don't
  ship the ambiguity.
- **Not checking your negatives.** Covered in §4.2, worth repeating because
  it's the easiest one to skip under time pressure: an unverified "this
  should be unanswerable" question can be silently wrong, and you'd never
  notice because a wrong refusal-ground-truth entry doesn't crash anything —
  it just quietly poisons one metric.
- **Ignoring §5's coupling trap.** The single most common way a team ends up
  confidently quoting a number that hasn't been true for months.

## 9. Cheat sheet

- Ground truth = a human-verified answer key, written **before** the system
  is tested against it.
- Five shapes: retrieval (which source), fact-fragment (a required string),
  full reference answer (the complete correct text), refusal/verdict
  (should this be declined), graded relevance (how good, not just
  right/wrong).
- Author it by **reading the real source**, not by guessing — for both what
  should be answerable and what genuinely shouldn't be.
- The hard negative (topic present, specific fact absent) tests more than
  the easy one (totally unrelated) — include both, weight toward the hard
  one.
- Match the ground truth shape to what your metric actually needs — don't
  build full reference answers if a fact fragment already answers your
  question.
- A score is meaningless without the exact system, version and date it was
  measured against stated alongside it.
- When the system changes, the ground truth needs re-checking — not just
  re-running.
