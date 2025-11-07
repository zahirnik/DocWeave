# packages/core/storage.py
"""
Storage utilities — safe local disk and optional S3 backend, tutorial-clear.

What this module provides
-------------------------
- Atomic local writes with path traversal defense.
- Simple MIME/type checks and size guards.
- Pluggable backends with a shared, tiny interface:
    Storage.save_bytes(path, data, content_type=None) -> str (return canonical path/key)
    Storage.open_bytes(path) -> bytes
    Storage.exists(path) -> bool
    Storage.delete(path) -> bool
    Storage.url_for(path, expires_s=900) -> Optional[str]   # signed URL when supported

Backends
--------
- LocalStorage (default): under DATA_DIR (e.g., ./data/uploads)
- S3Storage (optional):   when STORAGE_BACKEND=s3 and S3_* env vars are set

Environment
-----------
STORAGE_BACKEND=local|s3              (default: local)
DATA_DIR=./data                       (root for local storage)
STORAGE_ROOT=uploads                  (relative to DATA_DIR)
MAX_FILE_MB=64                        (coarse guard)
# S3:
S3_BUCKET=your-bucket
S3_REGION=us-east-1
S3_ENDPOINT_URL=                      (optional, for MinIO/LocalStack)
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...

Examples
--------
from packages.core.storage import get_storage, safe_join

st = get_storage()
dst = safe_join("t0", "job_123", "report.pdf")  # never allow path traversal
st.save_bytes(dst, b"...pdf bytes...", content_type="application/pdf")
print(st.url_for(dst))  # signed URL when on S3, else None
"""

from __future__ import annotations

import io
import mimetypes
import os
import tempfile
import shutil
from dataclasses import dataclass
from typing import Optional

from packages.core.config import get_settings
from packages.core.logging import get_logger

log = get_logger(__name__)

# ---------------------------
# Helpers / guards
# ---------------------------

_ALLOWED_MIME_PREFIXES = (
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/json",
    "application/vnd.openxmlformats-officedocument",  # xlsx/docx
    "image/png",
    "image/jpeg",
    "image/tiff",
)
_MAX_MB = int(os.getenv("MAX_FILE_MB", "64"))


def guess_mime(filename: str, default: str = "application/octet-stream") -> str:
    """
    Guess a content-type from a file extension; fall back to application/octet-stream.
    """
    ctype, _ = mimetypes.guess_type(filename)
    return ctype or default


def _rejects_mime(ctype: Optional[str]) -> bool:
    if not ctype:
        return False
    return not any(ctype.startswith(prefix) for prefix in _ALLOWED_MIME_PREFIXES)


def safe_join(*parts: str) -> str:
    """
    Join path fragments and **reject** any path traversal attempts.

    Returns a normalized relative path (posix-like). The storage backend will
    join this underneath its own root/bucket safely.

    Raises:
        ValueError on traversal (.. or absolute segments).
    """
    p = os.path.join(*[str(x).strip().lstrip("/\\") for x in parts if x])
    norm = os.path.normpath(p)
    if norm.startswith("..") or os.path.isabs(norm):
        raise ValueError("unsafe path (path traversal detected)")
    return norm.replace("\\", "/")


def atomic_write(path: str, data: bytes) -> None:
    """
    Atomic write helper for local files.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def _size_ok(n_bytes: int) -> bool:
    return n_bytes <= _MAX_MB * 1024 * 1024


# ---------------------------
# Interfaces
# ---------------------------

class Storage:
    """Abstract storage interface."""

    def save_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> str:
        raise NotImplementedError

    def open_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def delete(self, path: str) -> bool:
        raise NotImplementedError

    def url_for(self, path: str, expires_s: int = 900) -> Optional[str]:
        """Signed URL when supported (S3). Local storage returns None."""
        return None


# ---------------------------
# Local storage backend
# ---------------------------

@dataclass
class LocalStorage(Storage):
    root_dir: str

    def __post_init__(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)

    def _abs(self, rel: str) -> str:
        path = safe_join(rel)  # re-normalize each call
        return os.path.join(self.root_dir, path)

    def save_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> str:
        if not _size_ok(len(data)):
            raise ValueError(f"file too large: >{_MAX_MB}MB")
        if _rejects_mime(content_type or guess_mime(path)):
            raise ValueError(f"unsupported content-type: {content_type or guess_mime(path)}")
        abs_path = self._abs(path)
        atomic_write(abs_path, data)
        return path

    def open_bytes(self, path: str) -> bytes:
        abs_path = self._abs(path)
        with open(abs_path, "rb") as f:
            return f.read()

    def exists(self, path: str) -> bool:
        return os.path.exists(self._abs(path))

    def delete(self, path: str) -> bool:
        abs_path = self._abs(path)
        try:
            os.remove(abs_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def url_for(self, path: str, expires_s: int = 900) -> Optional[str]:
        # No public server is assumed in this scaffold; return None.
        # If you serve /data/uploads over HTTP, build a URL here.
        return None


# ---------------------------
# S3 storage backend (optional)
# ---------------------------

class S3Storage(Storage):
    def __init__(self, bucket: str, region: Optional[str] = None, endpoint_url: Optional[str] = None):
        try:
            import boto3  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("boto3 not installed; `pip install boto3`") from e

        self._boto3 = boto3
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url

        # Client with env credentials or shared config
        self._s3 = boto3.client(
            "s3",
            region_name=region or None,
            endpoint_url=endpoint_url or None,
            aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID") or None,
            aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY") or None,
        )

        # Light health check
        try:
            self._s3.head_bucket(Bucket=bucket)
        except Exception as e:  # pragma: no cover
            log.warning("S3: bucket %s head failed (%s). Ensure it exists and credentials are valid.", bucket, e)

    def _key(self, rel: str) -> str:
        return safe_join(rel)

    def save_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> str:
        if not _size_ok(len(data)):
            raise ValueError(f"file too large: >{_MAX_MB}MB")
        ctype = content_type or guess_mime(path)
        if _rejects_mime(ctype):
            raise ValueError(f"unsupported content-type: {ctype}")

        key = self._key(path)
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=ctype)
        return path

    def open_bytes(self, path: str) -> bytes:
        key = self._key(path)
        obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def exists(self, path: str) -> bool:
        key = self._key(path)
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, path: str) -> bool:
        key = self._key(path)
        try:
            self._s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def url_for(self, path: str, expires_s: int = 900) -> Optional[str]:
        """
        Return a presigned URL (GET) valid for `expires_s` seconds.
        """
        key = self._key(path)
        try:
            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=int(expires_s),
            )
        except Exception:
            return None


# ---------------------------
# Factory
# ---------------------------

_storage_singleton: Optional[Storage] = None


def get_storage() -> Storage:
    """
    Return a process-wide Storage instance according to env/settings.

    - local (default): LocalStorage under DATA_DIR / STORAGE_ROOT
    - s3:              S3Storage with provided bucket/region/endpoint
    """
    global _storage_singleton
    if _storage_singleton:
        return _storage_singleton

    st = get_settings()
    backend = (os.getenv("STORAGE_BACKEND") or "local").lower()
    root_rel = os.getenv("STORAGE_ROOT", "uploads").strip().strip("/")

    if backend == "s3":
        bucket = os.getenv("S3_BUCKET")
        if not bucket:
            log.warning("S3 backend requested but S3_BUCKET is not set; falling back to local.")
        else:
            _storage_singleton = S3Storage(
                bucket=bucket,
                region=os.getenv("S3_REGION"),
                endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
            )
            log.info("Storage: using S3 bucket=%s region=%s endpoint=%s", bucket, os.getenv("S3_REGION"), os.getenv("S3_ENDPOINT_URL"))
            return _storage_singleton

    # Local fallback
    local_root = os.path.join(st.data_dir, root_rel)
    _storage_singleton = LocalStorage(root_dir=local_root)
    log.info("Storage: using local root=%s", local_root)
    return _storage_singleton
