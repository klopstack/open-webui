"""Tests for the Qdrant BM25 hybrid search implementation.

These exercise the *real* backend modules (loaded via the ``conftest.py``
bootstrap) against a fake ``qdrant-client`` library client, covering:

* the pgvector-style contract (``hybrid_search`` override + automatic
  ``supports_hybrid_search`` detection, unchanged ``search`` signature),
* dense + sparse (BM25) fusion via ``merge_hybrid_search_results`` (RRF),
* graceful degradation (feature disabled, ``fastembed`` missing, non-hybrid
  collections, empty queries),
* named-vector addressing for hybrid collections,
* sparse-vector attachment on upsert,
* tenant scoping in the multitenancy client.
"""

import inspect

import pytest

import open_webui.retrieval.vector.dbs.qdrant as q
import open_webui.retrieval.vector.dbs.qdrant_multitenancy as qm
from open_webui.retrieval.vector.main import VectorDBBase
from open_webui.retrieval.vector.utils import merge_hybrid_search_results


def pt(pid, text, metadata=None, score=0.0):
    from conftest import _point

    return _point(pid, text, metadata or {}, score)


def rrf(weight, rank):
    """Mirror of the RRF score used by ``merge_hybrid_search_results``."""
    return weight / (60.0 + rank)


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
class TestContract:
    def test_both_clients_override_hybrid_search(self):
        assert q.QdrantClient.hybrid_search is not VectorDBBase.hybrid_search
        assert qm.QdrantClient.hybrid_search is not VectorDBBase.hybrid_search

    def test_base_hybrid_search_returns_none(self):
        class DummyBackend(VectorDBBase):
            pass

        DummyBackend.__abstractmethods__ = frozenset()
        assert VectorDBBase.hybrid_search(DummyBackend(), "c", "q", [[]]) is None

    def test_supports_hybrid_search_detection_predicate(self):
        # Mirrors AsyncVectorDBClient.supports_hybrid_search.
        class DummyBackend(VectorDBBase):
            pass

        DummyBackend.__abstractmethods__ = frozenset()
        assert type(DummyBackend()).hybrid_search is VectorDBBase.hybrid_search
        assert type(q.QdrantClient()).hybrid_search is not VectorDBBase.hybrid_search
        assert type(qm.QdrantClient()).hybrid_search is not VectorDBBase.hybrid_search

    def test_search_signature_has_no_legacy_query_param(self):
        # Regression guard: the original patch threaded a `query` param through
        # search(); the pgvector contract keeps search() untouched.
        for cls in (q.QdrantClient, qm.QdrantClient):
            params = inspect.signature(cls.search).parameters
            assert "query" not in params, f"{cls.__name__}.search gained a legacy 'query' param"

    def test_hybrid_search_signature_matches_contract(self):
        expected = {"collection_name", "query", "vectors", "filter", "limit", "hybrid_bm25_weight"}
        for cls in (q.QdrantClient, qm.QdrantClient):
            params = inspect.signature(cls.hybrid_search).parameters
            assert expected.issubset(params.keys()), f"{cls.__name__}.hybrid_search signature drifted"


# --------------------------------------------------------------------------- #
# Single-tenant client (qdrant.py)
# --------------------------------------------------------------------------- #
class TestSingleTenant:
    PREFIX = "open-webui"

    def _seed(self, fake, name, sparse):
        fake.collections[f"{self.PREFIX}_{name}"] = {"sparse": sparse}

    def test_disabled_returns_none(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=False, encoder=encoder)
        self._seed(fake, "c1", True)
        assert client.hybrid_search("c1", "query", [[0.1, 0.2]]) is None

    def test_fastembed_missing_degrades_to_none(self, make_client, monkeypatch):
        monkeypatch.setattr(q, "FASTEMBED_AVAILABLE", False)
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=None)
        self._seed(fake, "c1", True)
        assert client.hybrid_search("c1", "query", [[0.1, 0.2]]) is None

    @pytest.mark.parametrize("query", ["", "   ", None])
    def test_empty_query_returns_none(self, make_client, encoder, query):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", True)
        assert client.hybrid_search("c1", query, [[0.1, 0.2]]) is None

    def test_missing_collection_returns_none(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        assert client.hybrid_search("absent", "query", [[0.1, 0.2]]) is None

    def test_legacy_dense_only_collection_returns_none(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", sparse=False)
        assert client.hybrid_search("c1", "query", [[0.1, 0.2]]) is None

    def test_fuses_dense_and_sparse_with_rrf(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", True)
        fake.dense_points = [pt("A", "alpha", {"src": "dense-A"}, 0.9), pt("B", "beta", {"src": "dense-B"}, 0.8), pt("C", "gamma", {}, 0.7)]
        fake.sparse_points = [pt("B", "beta", {"src": "sparse-B"}, 0.5), pt("D", "delta", {}, 0.4), pt("A", "alpha", {}, 0.3)]

        result = client.hybrid_search("c1", "query", [[0.1, 0.2]], limit=10, hybrid_bm25_weight=0.5)

        # B leads: rank2 dense + rank1 sparse beats A's rank1 dense + rank3 sparse.
        assert result.ids[0] == ["B", "A", "D", "C"]
        assert result.documents[0][0] == "beta"
        # Overlapping doc keeps the dense leg's metadata (vector leg merges first).
        assert result.metadatas[0][0] == {"src": "dense-B"}
        distances = result.distances[0]
        assert distances == sorted(distances, reverse=True)
        assert abs(distances[0] - (rrf(0.5, 2) + rrf(0.5, 1))) < 1e-12

    def test_full_bm25_weight_skips_dense_leg(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", True)
        fake.dense_points = [pt("A", "alpha", {}, 0.9)]
        fake.sparse_points = [pt("B", "beta", {}, 0.5), pt("D", "delta", {}, 0.4), pt("A", "alpha", {}, 0.3)]

        result = client.hybrid_search("c1", "query", [[0.1, 0.2]], limit=10, hybrid_bm25_weight=1.0)

        assert result.ids[0] == ["B", "D", "A"]
        assert all(call.get("using") == "sparse" for call in fake.query_calls)

    def test_zero_bm25_weight_skips_sparse_leg(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", True)
        fake.dense_points = [pt("A", "alpha", {}, 0.9), pt("B", "beta", {}, 0.8), pt("C", "gamma", {}, 0.7)]
        fake.sparse_points = [pt("Z", "zeta", {}, 0.5)]

        result = client.hybrid_search("c1", "query", [[0.1, 0.2]], limit=10, hybrid_bm25_weight=0.0)

        assert result.ids[0] == ["A", "B", "C"]
        assert all(call.get("using") != "sparse" for call in fake.query_calls)

    def test_metadata_filter_applied_to_both_legs(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", True)
        fake.dense_points = [pt("A", "alpha", {}, 0.9)]
        fake.sparse_points = [pt("B", "beta", {}, 0.5)]

        client.hybrid_search("c1", "query", [[0.1, 0.2]], filter={"file_id": "f1"}, limit=5)

        for call in fake.query_calls:
            keys = [cond.key for cond in call["query_filter"].must]
            assert "metadata.file_id" in keys, f"filter not applied to leg {call.get('using') or 'dense'}"

    def test_search_addresses_named_dense_vector_on_hybrid_collection(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", True)
        fake.dense_points = [pt("A", "alpha", {}, 0.9)]

        client.search("c1", [[0.1, 0.2]], limit=3)

        assert fake.query_calls[-1]["query"] == ("dense", [0.1, 0.2])

    def test_search_uses_plain_vector_on_legacy_collection(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", sparse=False)
        fake.dense_points = [pt("A", "alpha", {}, 0.9)]

        client.search("c1", [[0.1, 0.2]], limit=3)

        assert fake.query_calls[-1]["query"] == [0.1, 0.2]

    def test_create_points_hybrid_attaches_sparse(self, make_client, encoder):
        client, _ = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        points = client._create_points([{"id": "1", "text": "hello world", "vector": [0.1, 0.2], "metadata": {"a": 1}}], collection_is_hybrid=True)
        assert set(points[0].vector.keys()) == {"dense", "sparse"}
        assert points[0].vector["dense"] == [0.1, 0.2]
        assert points[0].payload["text"] == "hello world"
        assert points[0].payload["metadata"] == {"a": 1}

    def test_create_points_hybrid_without_encoder_dense_named_only(self, make_client):
        client, _ = make_client(q.QdrantClient, hybrid=False, encoder=None)
        points = client._create_points([{"id": "1", "text": "hi", "vector": [0.1], "metadata": {}}], collection_is_hybrid=True)
        assert points[0].vector == {"dense": [0.1]}

    def test_create_points_legacy_plain_vector(self, make_client, encoder):
        client, _ = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        points = client._create_points([{"id": "1", "text": "hi", "vector": [0.1], "metadata": {}}], collection_is_hybrid=False)
        assert points[0].vector == [0.1]

    def test_upsert_detects_hybrid_collection_and_stores_sparse(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, "c1", True)

        client.upsert("c1", [{"id": "1", "text": "hello", "vector": [0.1, 0.2], "metadata": {}}])

        collection, points = fake.uploaded[-1]
        assert collection == f"{self.PREFIX}_c1"
        assert set(points[0].vector.keys()) == {"dense", "sparse"}

    def test_create_collection_hybrid_schema(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=True, encoder=encoder)
        client._create_collection("c1", 8)
        created = fake.created[-1]
        assert set(created["vectors_config"].keys()) == {"dense"}
        assert set(created["sparse_vectors_config"].keys()) == {"sparse"}

    def test_create_collection_legacy_schema(self, make_client, encoder):
        client, fake = make_client(q.QdrantClient, hybrid=False, encoder=None)
        client._create_collection("c1", 8)
        created = fake.created[-1]
        assert "sparse_vectors_config" not in created
        assert not isinstance(created["vectors_config"], dict)


# --------------------------------------------------------------------------- #
# Multitenancy client (qdrant_multitenancy.py)
# --------------------------------------------------------------------------- #
class TestMultitenancy:
    MEM = "open-webui_memories"

    def _seed(self, fake, sparse):
        fake.collections[self.MEM] = {"sparse": sparse}

    def test_tenant_mapping(self, make_client, encoder):
        client, _ = make_client(qm.QdrantClient, hybrid=True, encoder=encoder)
        assert client._get_collection_and_tenant_id("user-memory-alice") == (self.MEM, "user-memory-alice")
        assert client._get_collection_and_tenant_id("file-xyz") == ("open-webui_files", "file-xyz")

    def test_disabled_returns_none(self, make_client, encoder):
        client, fake = make_client(qm.QdrantClient, hybrid=False, encoder=encoder)
        self._seed(fake, True)
        assert client.hybrid_search("user-memory-alice", "query", [[0.1, 0.2]]) is None

    def test_fastembed_missing_degrades_to_none(self, make_client, monkeypatch):
        monkeypatch.setattr(qm, "FASTEMBED_AVAILABLE", False)
        client, fake = make_client(qm.QdrantClient, hybrid=True, encoder=None)
        self._seed(fake, True)
        assert client.hybrid_search("user-memory-alice", "query", [[0.1, 0.2]]) is None

    def test_legacy_collection_returns_none(self, make_client, encoder):
        client, fake = make_client(qm.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, sparse=False)
        assert client.hybrid_search("user-memory-alice", "query", [[0.1, 0.2]]) is None

    def test_fuses_with_tenant_scope(self, make_client, encoder):
        client, fake = make_client(qm.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, True)
        fake.dense_points = [pt("A", "alpha", {}, 0.9), pt("B", "beta", {}, 0.8)]
        fake.sparse_points = [pt("B", "beta", {}, 0.5), pt("C", "gamma", {}, 0.4)]

        result = client.hybrid_search("user-memory-alice", "query", [[0.1, 0.2]], limit=10, hybrid_bm25_weight=0.5)

        assert result.ids[0] == ["B", "A", "C"]
        # Every leg must be scoped to the tenant.
        for call in fake.query_calls:
            tenant_conds = [c for c in call["query_filter"].must if c.key == "tenant_id"]
            assert tenant_conds, f"leg {call.get('using') or 'dense'} missing tenant filter"
            assert tenant_conds[0].match.value == "user-memory-alice"

    def test_metadata_filter_propagates(self, make_client, encoder):
        client, fake = make_client(qm.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, True)
        fake.dense_points = [pt("A", "alpha", {}, 0.9)]
        fake.sparse_points = [pt("B", "beta", {}, 0.5)]

        client.hybrid_search("user-memory-alice", "query", [[0.1, 0.2]], filter={"file_id": "f1"}, limit=5)

        for call in fake.query_calls:
            keys = [c.key for c in call["query_filter"].must]
            assert "metadata.file_id" in keys

    def test_search_addresses_named_dense_vector_on_hybrid_collection(self, make_client, encoder):
        client, fake = make_client(qm.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, True)
        fake.dense_points = [pt("A", "alpha", {}, 0.9)]

        client.search("user-memory-alice", [[0.1, 0.2]], limit=3)

        assert fake.query_calls[-1]["query"] == ("dense", [0.1, 0.2])

    def test_upsert_stores_sparse_for_hybrid_collection(self, make_client, encoder):
        client, fake = make_client(qm.QdrantClient, hybrid=True, encoder=encoder)
        self._seed(fake, True)

        client.upsert("user-memory-alice", [{"id": "1", "text": "hello", "vector": [0.1, 0.2], "metadata": {}}])

        collection, points = fake.uploaded[-1]
        assert collection == self.MEM
        assert set(points[0].vector.keys()) == {"dense", "sparse"}
        assert points[0].payload["tenant_id"] == "user-memory-alice"

    def test_create_collection_hybrid_schema(self, make_client, encoder):
        client, fake = make_client(qm.QdrantClient, hybrid=True, encoder=encoder)
        client._create_multi_tenant_collection(self.MEM, 8)
        created = fake.created[-1]
        assert set(created["vectors_config"].keys()) == {"dense"}
        assert set(created["sparse_vectors_config"].keys()) == {"sparse"}


# --------------------------------------------------------------------------- #
# merge_hybrid_search_results (shared fusion primitive)
# --------------------------------------------------------------------------- #
class TestMergeHybridSearchResults:
    def test_dedupes_and_ranks_by_combined_score(self):
        from open_webui.retrieval.vector.main import SearchResult

        vector_result = SearchResult(
            ids=[["A", "B"]],
            distances=[[0.9, 0.8]],
            documents=[["alpha", "beta"]],
            metadatas=[[{"o": "A"}, {"o": "B"}]],
        )
        fts_results = [{"id": "B", "text": "beta", "vmetadata": {"o": "B-fts"}}, {"id": "C", "text": "gamma", "vmetadata": {}}]

        result = merge_hybrid_search_results(vector_result, fts_results, num_queries=1, limit=10, hybrid_bm25_weight=0.5)

        assert result.ids[0] == ["B", "A", "C"]
        assert result.metadatas[0][0] == {"o": "B"}  # vector leg wins for overlapping ids
        assert result.documents[0][2] == "gamma"

    def test_weights_clamped(self):
        from open_webui.retrieval.vector.main import SearchResult

        vector_result = SearchResult(ids=[["A"]], distances=[[0.9]], documents=[["alpha"]], metadatas=[[{}]])
        high = merge_hybrid_search_results(vector_result, [], num_queries=1, limit=10, hybrid_bm25_weight=5.0)
        clamped = merge_hybrid_search_results(vector_result, [], num_queries=1, limit=10, hybrid_bm25_weight=1.0)
        assert high.distances[0] == clamped.distances[0]
