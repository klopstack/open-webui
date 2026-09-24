"""Test bootstrap for the Qdrant hybrid-search unit tests.

Importing the real Qdrant backends normally drags in the full
``open_webui.config`` -> ``open_webui.env`` / ORM dependency tree (plus
``typer``, ``uvicorn``, ``redis``, ``authlib``). None of that is relevant to the
logic under test, so we register lightweight stand-ins for the handful of
modules the backends import, then load the *real* backend modules so the tests
exercise genuine production code paths (``hybrid_search``, ``search``,
``_create_points``, ``merge_hybrid_search_results``).

Only ``qdrant-client`` (a real, pinned dependency) is required; ``fastembed`` is
deliberately *not* required so the graceful-degradation paths can be exercised.
"""

import sys
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]  # .../backend
OPEN_WEBUI_DIR = BACKEND_DIR / "open_webui"


def _bootstrap() -> None:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    # Top-level package: bypass the heavy __init__ (typer/uvicorn CLI).
    if "open_webui" not in sys.modules:
        pkg = types.ModuleType("open_webui")
        pkg.__path__ = [str(OPEN_WEBUI_DIR)]
        sys.modules["open_webui"] = pkg

    # open_webui.config: only the QDRANT_* constants the backends import.
    if "open_webui.config" not in sys.modules:
        cfg = types.ModuleType("open_webui.config")
        for name, default in {
            "QDRANT_API_KEY": None,
            "QDRANT_COLLECTION_PREFIX": "open-webui",
            "QDRANT_DENSE_VECTOR_NAME": "dense",
            "QDRANT_GRPC_PORT": 6334,
            "QDRANT_HNSW_M": 16,
            "QDRANT_HYBRID_SEARCH_ENABLED": False,
            "QDRANT_ON_DISK": False,
            "QDRANT_PREFER_GRPC": False,
            "QDRANT_SPARSE_EMBEDDING_MODEL": "Qdrant/bm25",
            "QDRANT_SPARSE_ON_DISK": False,
            "QDRANT_SPARSE_VECTOR_NAME": "sparse",
            "QDRANT_TIMEOUT": 5,
            "QDRANT_URI": "http://localhost:6333",
        }.items():
            setattr(cfg, name, default)
        sys.modules["open_webui.config"] = cfg

    # open_webui.env: only the constant vector/utils.py reads.
    if "open_webui.env" not in sys.modules:
        env = types.ModuleType("open_webui.env")
        env.RAG_METADATA_MAX_VALUE_CHARS = None
        sys.modules["open_webui.env"] = env

    # open_webui.utils.misc: only sanitize_text_for_db is used by vector/utils.
    if "open_webui.utils.misc" not in sys.modules:
        misc = types.ModuleType("open_webui.utils.misc")
        misc.sanitize_text_for_db = lambda value: value
        sys.modules["open_webui.utils.misc"] = misc


_bootstrap()


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Arr:
    """Minimal stand-in for a numpy array exposing ``.tolist()``."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _Emb:
    def __init__(self, indices, values):
        self.indices = _Arr(indices)
        self.values = _Arr(values)


class FakeSparseEncoder:
    """Deterministic stand-in for ``fastembed.sparse.SparseTextEmbedding``.

    The concrete sparse values are irrelevant to the fusion logic under test;
    we only need ``embed``/``query_embed`` to return objects shaped like
    fastembed's (``.indices`` / ``.values`` with ``.tolist()``).
    """

    def embed(self, texts):
        return [_Emb([ord(c) % 7 + 1 for c in t], [1.0] * max(1, len(t))) for t in texts]

    def query_embed(self, text):
        return self.embed([text])


class _Point:
    def __init__(self, id, payload, score=0.0):
        self.id = id
        self.payload = payload
        self.score = score


def _point(id, text, metadata=None, score=0.0):
    return _Point(id, {"text": text, "metadata": metadata or {}}, score)


class FakeQdrantLibClient:
    """Records calls and returns canned results for the qdrant-client API
    surface the backends use. Branches ``query_points`` on the ``using`` kwarg
    so the dense leg (no ``using``) and the sparse/BM25 leg (``using='sparse'``)
    can be served independently."""

    def __init__(self):
        self.collections = {}  # name -> {"sparse": bool}
        self.created = []
        self.payload_indexes = []
        self.uploaded = []  # (collection, [points])
        self.deleted = []
        self.query_calls = []  # recorded query_points kwargs
        self.dense_points = []
        self.sparse_points = []
        self.scroll_points = []

    # -- collection introspection ----------------------------------------- #
    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def get_collection(self, name):
        if name not in self.collections:
            raise KeyError(name)
        sparse = self.collections[name]["sparse"]
        params = types.SimpleNamespace(sparse_vectors=("sparse" if sparse else None))
        return types.SimpleNamespace(config=types.SimpleNamespace(params=params))

    def get_collections(self):
        return types.SimpleNamespace(
            collections=[types.SimpleNamespace(name=n) for n in self.collections]
        )

    # -- writes ----------------------------------------------------------- #
    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        name = kwargs["collection_name"]
        sparse_cfg = kwargs.get("sparse_vectors_config")
        self.collections[name] = {"sparse": bool(sparse_cfg)}
        return True

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs)

    def upload_points(self, collection_name, points):
        self.uploaded.append((collection_name, list(points)))

    def upsert(self, collection_name, points):
        self.uploaded.append((collection_name, list(points)))

    def delete(self, **kwargs):
        self.deleted.append(kwargs)

    # -- reads ------------------------------------------------------------ #
    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        points = self.sparse_points if kwargs.get("using") == "sparse" else self.dense_points
        return types.SimpleNamespace(points=list(points))

    def scroll(self, **kwargs):
        return (list(self.scroll_points), None)

    def count(self, **kwargs):
        return types.SimpleNamespace(count=len(self.scroll_points))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_client(monkeypatch):
    """Build a real OWUI Qdrant client wired to a fresh :class:`FakeQdrantLibClient`.

    Usage::

        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=FakeSparseEncoder())
    """
    import open_webui.retrieval.vector.dbs.qdrant as q
    import open_webui.retrieval.vector.dbs.qdrant_multitenancy as qm

    def _make(cls, *, hybrid=True, encoder=None):
        module = q if cls is q.QdrantClient else qm
        fake = FakeQdrantLibClient()
        monkeypatch.setattr(module, "Qclient", lambda *a, **k: fake)
        inst = cls()
        inst.client = fake
        inst.QDRANT_HYBRID_SEARCH_ENABLED = hybrid
        if encoder is not None:
            inst._sparse_encoder = encoder
        return inst, fake

    return _make


@pytest.fixture
def encoder():
    return FakeSparseEncoder()
