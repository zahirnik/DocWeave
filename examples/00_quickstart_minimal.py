# examples/00_quickstart_minimal.py
"""
Quickstart (10–20 lines): load a few texts, build a tiny in-memory index,
ask a question, and print the top match with a cosine score.

Uses the Embeddings facade (no DB/vectorstore). If OPENAI_API_KEY is set, it uses
OpenAI embeddings; otherwise it tries a local model via sentence-transformers.
"""

from __future__ import annotations

import math  # kept to preserve original structure
import os
import numpy as np

from packages.core.config import get_settings
# [CHANGED] Use Embeddings + EmbeddingConfig (EmbeddingClient no longer exists)
from packages.retriever.embeddings import Embeddings, EmbeddingConfig  # [CHANGED]

# 1) Config & embedder
cfg = get_settings()
# [CHANGED] Auto-select provider: OpenAI if key exists; else local model
use_openai = bool(os.getenv("OPENAI_API_KEY"))
if use_openai:
    embed_cfg = EmbeddingConfig(provider="openai", model="text-embedding-3-small", normalize=True)  # [CHANGED]
else:
    embed_cfg = EmbeddingConfig(provider="local", model="BAAI/bge-small-en-v1.5", normalize=True)   # [CHANGED]
embedder = Embeddings(embed_cfg)  # [CHANGED]

# 2) Small corpus (inline for demo; you can swap with ./data/samples/*.txt)
docs = [
    {"id": "acme-2024-overview", "text": "ACME PLC reported a 38.2% gross margin in FY2024 with strong cost control."},
    {"id": "beta-q3-rev",         "text": "Beta Corp revenue in Q3 2024 was $1.26 billion, up 14.5% year over year."},
    {"id": "gamma-expenses",      "text": "Gamma Ltd major expenses were cost of sales, SG&A, and R&D in 2024."},
]

# 3) Build an in-memory index: E = unit-norm embeddings
# [CHANGED] .embed(...) -> .embed_documents(...)
doc_embeddings = np.array(embedder.embed_documents([d["text"] for d in docs]), dtype="float32")  # [CHANGED]
doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12

# 4) Query → embedding → cosine similarity
question = "What was ACME's gross margin in 2024?"
# [CHANGED] .embed([q])[0] -> .embed_query(q)
q = np.array(embedder.embed_query(question), dtype="float32")  # [CHANGED]
q /= (np.linalg.norm(q) + 1e-12)

scores = (doc_embeddings @ q)  # cosine because both are unit-norm
top_i = int(np.argmax(scores))
print(f"Q: {question}")
print(f"Top doc: {docs[top_i]['id']} | score={float(scores[top_i]):.3f}")
print(f"Text: {docs[top_i]['text']}")
