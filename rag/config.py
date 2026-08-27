"""Central configuration. Everything tunable lives here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
INDEX_DIR = DATA / "index"

# --- Models -----------------------------------------------------------------
# Embedding model. Two supported, with different capabilities:
#   gemini-embedding-001 : text-only, 2048 input tokens, supports `task_type`
#                          (asymmetric query/document embeddings). GA.
#   gemini-embedding-2   : multimodal, 8192 input tokens, NO task_type,
#                          auto-normalizes truncated dimensions. GA, newest.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001")

# Matryoshka (MRL) truncation. Native is 3072; 768/1536/3072 are recommended.
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1536"))

# gemini-2.5-flash is listed by the API but closed to new keys.
#
# flash-lite was the default until it was caught over-refusing: asked "what is
# the best X", with the subject plainly in the context at cosine 0.79, it
# emitted the refusal sentinel instead of answering what the documents do say.
# It obeys rules 1-3 of the system prompt but not the subtler 4-6. Over-refusal
# is the worst failure mode here, because the system looks correctly grounded
# while being useless, and nothing in the output signals that it went wrong.
# 3.6-flash and 3.5-flash both handle it, so correctness wins over quota.
#
# Set RAG_GEN_MODEL=gemini-3.5-flash-lite only for high-volume runs where the
# tighter daily quota on flash actually bites, and expect more refusals.
GEN_MODEL = os.getenv("RAG_GEN_MODEL", "gemini-3.6-flash")

API_KEY = os.getenv("GEMINI_API_KEY")

# --- Chunking ---------------------------------------------------------------
CHUNK_CHARS = 1400        # target chunk size in characters (~350 tokens)
CHUNK_OVERLAP = 200       # sliding overlap so facts aren't cut in half
MIN_CHUNK_CHARS = 120     # drop fragments smaller than this

# --- Embedding requests -----------------------------------------------------
EMBED_BATCH = 32          # texts per API call
EMBED_MAX_RETRIES = 6

# The free tier counts *individual texts*, not HTTP calls: a batch of 32
# spends 32 units against a 100-per-minute quota. Stay just under it rather
# than discovering the ceiling by being rejected.
EMBED_ITEMS_PER_MIN = int(os.getenv("RAG_EMBED_RPM", "95"))

# Generation is limited per *minute* and, on the free tier, per *day* as well
# (20/day/model at time of writing). The per-minute cap is what this throttles;
# the daily one is surfaced as an error, because it cannot be waited out.
GEN_PER_MIN = int(os.getenv("RAG_GEN_RPM", "10"))

# --- Retrieval --------------------------------------------------------------
CANDIDATES = 30           # how many each retriever proposes before fusion
TOP_K = 6                 # chunks actually shown to the generator
RRF_K = 60                # reciprocal-rank-fusion smoothing constant
MMR_LAMBDA = 0.7          # 1.0 = pure relevance, 0.0 = pure diversity

# Below this fused-cosine score we assume the corpus has nothing relevant
# and refuse *before* spending a generation call.
MIN_COSINE = 0.55
