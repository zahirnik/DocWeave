# packages/retriever/embeddings.py
"""
Embedding utilities — OpenAI, AWS Bedrock, or local (Sentence-Transformers).

Public API
----------
- EmbeddingConfig
- Embeddings
- EmbeddingClient (compat wrapper exposing .embed([...]))
- embed_texts(texts: list[str]) -> list[list[float]]
- embed_text(text: str) -> list[float]

Providers
---------
- "auto"    : OPENAI if key, else BEDROCK if region/creds, else "local"
- "openai"  : OpenAI embeddings
- "bedrock" : AWS Bedrock (Titan/Cohere)
- "local"   : sentence-transformers

Env (all optional)
------------------
EMBEDDINGS_PROVIDER=auto|openai|bedrock|local
EMBEDDINGS_MODEL=...
EMBEDDINGS_DIM=...
EMBEDDINGS_BATCH=64
EMBEDDINGS_TIMEOUT_S=30
EMBEDDINGS_MAX_RETRIES=3
EMBEDDINGS_NORMALIZE=true|false
OPENAI_API_KEY=...
AWS_REGION / AWS_DEFAULT_REGION / BEDROCK_REGION
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union, Any

from packages.core.logging import get_logger

log = get_logger(__name__)

# ---------------------------
# Small helpers
# ---------------------------

def _batched(seq: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    n = len(seq)
    b = max(1, int(batch_size))
    for i in range(0, n, b):
        yield seq[i : i + b]

def _l2_normalize(v: Sequence[float]) -> List[float]:
    s = math.sqrt(sum((x * x) for x in v)) or 1.0
    return [float(x / s) for x in v]

def _json_dumps(obj: Any) -> bytes:
    import json
    return json.dumps(obj).encode("utf-8")

def _json_loads(buf: Union[str, bytes]) -> Any:
    import json
    if isinstance(buf, bytes):
        buf = buf.decode("utf-8")
    return json.loads(buf)

def _resolve_auto_provider() -> str:
    """Resolve 'auto' provider based on environment."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
        return "bedrock"
    return "local"

# ---------------------------
# Config (env-only; no get_settings dependency)
# ---------------------------

@dataclass
class EmbeddingConfig:
    provider: str = "auto"                   # "auto" | "openai" | "bedrock" | "local"
    model: str = "text-embedding-3-small"    # default varies by provider
    dim: Optional[int] = None
    batch_size: int = 64
    normalize: bool = True
    timeout_s: int = 30
    max_retries: int = 3
    aws_region: Optional[str] = None
    bedrock_region: Optional[str] = None

    @staticmethod
    def from_env() -> "EmbeddingConfig":
        provider = (os.getenv("EMBEDDINGS_PROVIDER", "auto") or "auto").strip().lower()

        # Resolve 'auto' here using env
        if provider == "auto":
            provider = _resolve_auto_provider()

        default_model = (
            "text-embedding-3-small" if provider == "openai"
            else ("amazon.titan-embed-text-v2:0" if provider == "bedrock"
                  else "BAAI/bge-small-en-v1.5")
        )
        model = os.getenv("EMBEDDINGS_MODEL", default_model) or default_model

        dim_env = (os.getenv("EMBEDDINGS_DIM") or "").strip()
        dim = int(dim_env) if dim_env.isdigit() else None

        batch = int(os.getenv("EMBEDDINGS_BATCH", "64"))
        timeout_s = int(os.getenv("EMBEDDINGS_TIMEOUT_S", "30"))
        max_retries = int(os.getenv("EMBEDDINGS_MAX_RETRIES", "3"))

        norm_env = os.getenv("EMBEDDINGS_NORMALIZE", "true").lower()
        normalize = norm_env in {"1", "true", "yes", "on"}

        aws_region = os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        bedrock_region = os.getenv("BEDROCK_REGION")

        return EmbeddingConfig(
            provider=provider,
            model=model,
            dim=dim,
            batch_size=batch,
            normalize=normalize,
            timeout_s=timeout_s,
            max_retries=max_retries,
            aws_region=aws_region,
            bedrock_region=bedrock_region,
        )

# ---------------------------
# Facade
# ---------------------------

class Embeddings:
    """
    Small, explicit embedding facade.
    """
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.cfg = config or EmbeddingConfig.from_env()

        # Copy config fields to instance vars
        self._provider = (self.cfg.provider or "auto").strip().lower()
        self._model = self.cfg.model
        self._dim = self.cfg.dim
        self._batch = max(1, int(self.cfg.batch_size))
        self._normalize = bool(self.cfg.normalize)
        self._timeout_s = int(self.cfg.timeout_s)
        self._retries = int(self.cfg.max_retries)
        self._aws_region = self.cfg.bedrock_region or self.cfg.aws_region

        # Safety: resolve 'auto' here too (covers cases where caller forces provider='auto')
        if self._provider == "auto":
            self._provider = _resolve_auto_provider()

        log.info("Embeddings: provider=%s model=%s batch=%d normalize=%s",
                 self._provider, self._model, self._batch, self._normalize)

    # Public API
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        clean = [self._sanitize_text(t) for t in texts]
        vecs = self._embed_batch(clean)
        return [self._postprocess(v) for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    # Back-compat alias used by older examples
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    # Internals
    @staticmethod
    def _sanitize_text(x: object) -> str:
        s = "" if x is None else str(x)
        return s.replace("\r\n", "\n").strip()

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if self._provider == "openai":
            return self._embed_openai(texts)
        if self._provider == "bedrock":
            return self._embed_bedrock(texts)
        if self._provider == "local":
            return self._embed_local(texts)
        raise RuntimeError(f"Unknown embeddings provider: {self._provider}")

    def _postprocess(self, v: List[float]) -> List[float]:
        return _l2_normalize(v) if self._normalize else v

    # OpenAI
    def _embed_openai(self, texts: Sequence[str]) -> List[List[float]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use provider 'openai'.")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("`openai` package not installed. Run: pip install openai") from e

        client = OpenAI(api_key=api_key)
        model = self._model
        out: List[List[float]] = []

        for batch in _batched(texts, self._batch):
            last_err: Optional[Exception] = None
            for attempt in range(self._retries + 1):
                try:
                    # SDK v1; timeout passed via client config normally—kept here for simplicity
                    resp = client.embeddings.create(model=model, input=list(batch))
                    got = [d.embedding for d in resp.data]
                    if self._dim is None and got:
                        self._dim = len(got[0])
                        log.info("Embeddings: inferred dimension=%d from OpenAI model=%s", self._dim, model)
                    out.extend(got)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < self._retries:
                        sleep_s = min(2 ** attempt, 8)
                        log.info("OpenAI embed retry %d/%d after error: %s (sleep=%ss)",
                                 attempt + 1, self._retries, e, sleep_s)
                        time.sleep(sleep_s)
                    else:
                        raise RuntimeError(f"OpenAI embeddings failed after retries: {e}") from e
        return out

    # AWS Bedrock (Titan/Cohere)
    def _embed_bedrock(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            import boto3  # type: ignore
            from botocore.config import Config as BotoConfig  # type: ignore
        except Exception as e:
            raise RuntimeError("Bedrock provider requires boto3. Run: pip install boto3 botocore") from e

        region = self._aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        boto_cfg = BotoConfig(
            connect_timeout=self._timeout_s,
            read_timeout=self._timeout_s,
            retries={"max_attempts": self._retries + 1},
        )
        try:
            client = boto3.client("bedrock-runtime", region_name=region, config=boto_cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to create Bedrock client (check AWS credentials/region). Details: {e}") from e

        model_id = self._model
        out: List[List[float]] = []

        def _invoke(payload: dict) -> dict:
            resp = client.invoke_model(modelId=model_id, body=_json_dumps(payload))
            raw = resp.get("body")
            body_bytes = raw.read() if hasattr(raw, "read") else raw
            return _json_loads(body_bytes)

        for batch in _batched(texts, self._batch):
            batch = list(batch)
            last_err: Optional[Exception] = None
            for attempt in range(self._retries + 1):
                try:
                    if "titan-embed-text" in model_id:
                        for t in batch:
                            payload = {"inputText": t}
                            resp = _invoke(payload)
                            vec = resp.get("embedding") or resp.get("vector")
                            if not isinstance(vec, list):
                                raise RuntimeError(f"Unexpected Titan response: {resp}")
                            out.append([float(x) for x in vec])
                    elif "cohere" in model_id or "embed-" in model_id:
                        payload = {"texts": batch, "input_type": "search_document"}
                        resp = _invoke(payload)
                        vecs = resp.get("embeddings")
                        if not isinstance(vecs, list):
                            raise RuntimeError(f"Unexpected Cohere response: {resp}")
                        for v in vecs:
                            out.append([float(x) for x in v])
                    else:
                        raise RuntimeError(
                            f"Bedrock model '{model_id}' not recognized.\n"
                            f"Known: amazon.titan-embed-text-v2:0, cohere.embed-english-v3"
                        )
                    if self._dim is None and out:
                        self._dim = len(out[-1])
                        log.info("Embeddings: inferred dimension=%d from Bedrock model=%s", self._dim, model_id)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < self._retries:
                        sleep_s = min(2 ** attempt, 8)
                        log.info("Bedrock embed retry %d/%d after error: %s (sleep=%ss)",
                                 attempt + 1, self._retries, e, sleep_s)
                        time.sleep(sleep_s)
                    else:
                        raise RuntimeError(f"Bedrock embeddings failed after retries: {e}") from e
        return out

    # Local (sentence-transformers)
    def _embed_local(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("Local embeddings require `sentence-transformers`. Run: pip install sentence-transformers") from e

        model_name = self._model or "BAAI/bge-small-en-v1.5"
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load local embedding model '{model_name}': {e}") from e

        out: List[List[float]] = []
        for batch in _batched(texts, self._batch):
            vecs = model.encode(list(batch), normalize_embeddings=False, show_progress_bar=False)
            if hasattr(vecs, "tolist"):
                vecs = vecs.tolist()
            out.extend(vecs)

        if self._dim is None and out:
            self._dim = len(out[0])
            log.info("Embeddings: inferred dimension=%d from local model=%s", self._dim, model_name)
        return out

# ---------------------------
# Back-compat wrappers & helpers
# ---------------------------

_MODEL_ALIASES = {
    "mini": {
        "openai": "text-embedding-3-small",
        "bedrock": "amazon.titan-embed-text-v2:0",
        "local": "BAAI/bge-small-en-v1.5",
    },
    "large": {
        "openai": "text-embedding-3-large",
        "bedrock": "cohere.embed-english-v3",
        "local": "intfloat/e5-large-v2",
    },
}

def _resolve_model_alias(provider: str, alias_or_name: Optional[str]) -> Optional[str]:
    if not alias_or_name:
        return None
    alias = alias_or_name.strip().lower()
    if alias in _MODEL_ALIASES:
        prov = provider.strip().lower()
        prov = prov if prov in {"openai", "bedrock", "local"} else _resolve_auto_provider()
        return _MODEL_ALIASES[alias].get(prov)
    return alias_or_name  # treat as explicit model name

class EmbeddingClient:  # pragma: no cover
    """Back-compat wrapper exposing .embed([...]) like the old client."""
    def __init__(self, provider: Optional[str] = None, model_alias: Optional[str] = None, **kw):
        cfg = EmbeddingConfig.from_env()

        # Provider override: only override if NOT 'auto'
        if provider:
            p = provider.strip().lower()
            if p != "auto":
                cfg.provider = p
            else:
                # Keep resolved env-based provider
                cfg.provider = cfg.provider or _resolve_auto_provider()

        # Resolve model alias against provider
        resolved_model = _resolve_model_alias(cfg.provider, model_alias)
        if resolved_model:
            cfg.model = resolved_model

        # Other kwargs to config if present (e.g., batch_size=..., normalize=...)
        for k, v in kw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        self._enc = Embeddings(cfg)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self._enc.embed_documents(texts)

def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Convenience free function."""
    return Embeddings().embed_documents(texts)

def embed_text(text: str) -> List[float]:
    """Convenience free function."""
    return Embeddings().embed_query(text)

__all__ = ["EmbeddingConfig", "Embeddings", "EmbeddingClient", "embed_texts", "embed_text"]
