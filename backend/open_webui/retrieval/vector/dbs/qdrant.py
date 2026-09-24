"""
NOTE: This vector database integration is community-supported and maintained on a best-effort basis.
"""

import logging
from typing import Any, Optional
from urllib.parse import urlparse

from open_webui.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_PREFIX,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_GRPC_PORT,
    QDRANT_HNSW_M,
    QDRANT_HYBRID_SEARCH_ENABLED,
    QDRANT_ON_DISK,
    QDRANT_PREFER_GRPC,
    QDRANT_SPARSE_EMBEDDING_MODEL,
    QDRANT_SPARSE_ON_DISK,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_TIMEOUT,
    QDRANT_URI,
)
from open_webui.retrieval.vector.main import (
    GetResult,
    SearchResult,
    VectorDBBase,
    VectorItem,
)
from open_webui.retrieval.vector.utils import iter_filter_conditions, merge_hybrid_search_results, process_metadata
from qdrant_client import QdrantClient as Qclient
from qdrant_client.http.models import PointStruct, SparseVector
from qdrant_client.models import models

try:
    from fastembed.sparse import SparseTextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

NO_LIMIT = 999999999

log = logging.getLogger(__name__)


def _metadata_filter(key: str, op: str, value: Any) -> models.FieldCondition:
    match = models.MatchAny(any=value) if op == '$in' else models.MatchValue(value=value)
    return models.FieldCondition(key=f'metadata.{key}', match=match)


class QdrantClient(VectorDBBase):
    def __init__(self):
        self.collection_prefix = QDRANT_COLLECTION_PREFIX
        self.QDRANT_URI = QDRANT_URI
        self.QDRANT_API_KEY = QDRANT_API_KEY
        self.QDRANT_ON_DISK = QDRANT_ON_DISK
        self.PREFER_GRPC = QDRANT_PREFER_GRPC
        self.GRPC_PORT = QDRANT_GRPC_PORT
        self.QDRANT_TIMEOUT = QDRANT_TIMEOUT
        self.QDRANT_HNSW_M = QDRANT_HNSW_M
        self.QDRANT_HYBRID_SEARCH_ENABLED = QDRANT_HYBRID_SEARCH_ENABLED
        self.QDRANT_SPARSE_EMBEDDING_MODEL = QDRANT_SPARSE_EMBEDDING_MODEL
        self.QDRANT_DENSE_VECTOR_NAME = QDRANT_DENSE_VECTOR_NAME
        self.QDRANT_SPARSE_VECTOR_NAME = QDRANT_SPARSE_VECTOR_NAME
        self.QDRANT_SPARSE_ON_DISK = QDRANT_SPARSE_ON_DISK
        # Lazy-initialized on first use so that instantiating the client never
        # triggers a sparse model download.
        self._sparse_encoder = None
        self._sparse_encoder_warned = False

        if not self.QDRANT_URI:
            self.client = None
            return

        # Unified handling for either scheme
        parsed = urlparse(self.QDRANT_URI)
        host = parsed.hostname or self.QDRANT_URI
        http_port = parsed.port or 6333  # default REST port

        if self.PREFER_GRPC:
            self.client = Qclient(
                host=host,
                port=http_port,
                grpc_port=self.GRPC_PORT,
                prefer_grpc=self.PREFER_GRPC,
                api_key=self.QDRANT_API_KEY,
                timeout=self.QDRANT_TIMEOUT,
            )
        else:
            self.client = Qclient(
                url=self.QDRANT_URI,
                api_key=self.QDRANT_API_KEY,
                timeout=QDRANT_TIMEOUT,
            )

    def _get_sparse_encoder(self):
        """Return the fastembed sparse encoder, loading it lazily on first use.

        Loading is deferred so that merely instantiating the client (or importing
        the module) never triggers a model download. Returns None when hybrid
        search is disabled or the optional `fastembed` dependency is missing.
        """
        if not self.QDRANT_HYBRID_SEARCH_ENABLED:
            return None
        if self._sparse_encoder is None:
            if not FASTEMBED_AVAILABLE:
                if not self._sparse_encoder_warned:
                    log.warning(
                        "Qdrant hybrid search is enabled but 'fastembed' is not installed; "
                        "falling back to dense-only search. Install with: pip install 'fastembed>=0.6.1'"
                    )
                    self._sparse_encoder_warned = True
                return None
            try:
                self._sparse_encoder = SparseTextEmbedding(model_name=self.QDRANT_SPARSE_EMBEDDING_MODEL)
                log.info('Loaded Qdrant sparse embedding model: %s', self.QDRANT_SPARSE_EMBEDDING_MODEL)
            except Exception as e:
                log.warning('Failed to load Qdrant sparse embedding model (%s); falling back to dense-only search.', e)
                return None
        return self._sparse_encoder

    def _is_hybrid_collection(self, prefixed_collection_name: str) -> bool:
        """True if the collection stores sparse vectors (was created in hybrid mode)."""
        try:
            info = self.client.get_collection(prefixed_collection_name)
            return bool(info.config.params.sparse_vectors)
        except Exception:
            return False

    def _encode_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts into sparse vectors using the fastembed sparse model."""
        embeddings = list(self._get_sparse_encoder().embed(texts))
        return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embeddings]

    def _encode_sparse_query(self, text: str) -> SparseVector:
        """Encode a query text into a sparse vector."""
        embeddings = list(self._get_sparse_encoder().query_embed(text))
        e = embeddings[0]
        return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

    def _result_to_get_result(self, points) -> GetResult:
        ids = []
        documents = []
        metadatas = []

        for point in points:
            payload = point.payload
            ids.append(point.id)
            documents.append(payload['text'])
            metadatas.append(payload['metadata'])

        return GetResult(
            **{
                'ids': [ids],
                'documents': [documents],
                'metadatas': [metadatas],
            }
        )

    def _create_collection(self, collection_name: str, dimension: int):
        collection_name_with_prefix = f'{self.collection_prefix}_{collection_name}'
        if self.QDRANT_HYBRID_SEARCH_ENABLED:
            self.client.create_collection(
                collection_name=collection_name_with_prefix,
                vectors_config={
                    self.QDRANT_DENSE_VECTOR_NAME: models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                        on_disk=self.QDRANT_ON_DISK,
                    )
                },
                sparse_vectors_config={
                    self.QDRANT_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=self.QDRANT_SPARSE_ON_DISK)
                    )
                },
                hnsw_config=models.HnswConfigDiff(
                    m=self.QDRANT_HNSW_M,
                ),
            )
            log.info('Hybrid collection %s created with dense (%s dims) + sparse vectors', collection_name_with_prefix, dimension)
        else:
            self.client.create_collection(
                collection_name=collection_name_with_prefix,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=self.QDRANT_ON_DISK,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=self.QDRANT_HNSW_M,
                ),
            )

        # Create payload indexes for efficient filtering
        self.client.create_payload_index(
            collection_name=collection_name_with_prefix,
            field_name='metadata.hash',
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=False,
                on_disk=self.QDRANT_ON_DISK,
            ),
        )
        self.client.create_payload_index(
            collection_name=collection_name_with_prefix,
            field_name='metadata.file_id',
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=False,
                on_disk=self.QDRANT_ON_DISK,
            ),
        )
        log.info('collection %s successfully created!', collection_name_with_prefix)

    def _create_collection_if_not_exists(self, collection_name, dimension):
        if not self.has_collection(collection_name=collection_name):
            self._create_collection(collection_name=collection_name, dimension=dimension)

    def _create_points(self, items: list[VectorItem], collection_is_hybrid: bool = False) -> list[PointStruct]:
        sparse_vectors = None
        if collection_is_hybrid and self._get_sparse_encoder() is not None:
            sparse_vectors = self._encode_sparse([item['text'] for item in items])

        points = []
        for i, item in enumerate(items):
            if collection_is_hybrid:
                vector: Any = {self.QDRANT_DENSE_VECTOR_NAME: item['vector']}
                if sparse_vectors is not None:
                    vector[self.QDRANT_SPARSE_VECTOR_NAME] = sparse_vectors[i]
            else:
                vector = item['vector']
            points.append(
                PointStruct(
                    id=item['id'],
                    vector=vector,
                    payload={'text': item['text'], 'metadata': process_metadata(item['metadata'])},
                )
            )
        return points

    def has_collection(self, collection_name: str) -> bool:
        return self.client.collection_exists(f'{self.collection_prefix}_{collection_name}')

    def delete_collection(self, collection_name: str):
        return self.client.delete_collection(collection_name=f'{self.collection_prefix}_{collection_name}')

    def search(
        self,
        collection_name: str,
        vectors: list[list[float | int]],
        filter: Optional[dict] = None,
        limit: int = 10,
    ) -> Optional[SearchResult]:
        # Search for the nearest neighbor items based on the vectors and return 'limit' number of results.
        if limit is None:
            limit = NO_LIMIT  # otherwise qdrant would set limit to 10!

        conditions = [_metadata_filter(key, op, value) for key, op, value in iter_filter_conditions(filter)]
        query_filter = models.Filter(must=conditions) if conditions else None
        # Hybrid collections store the dense vector under a named key, so the
        # query must address it explicitly.
        prefixed_name = f'{self.collection_prefix}_{collection_name}'
        query = (self.QDRANT_DENSE_VECTOR_NAME, vectors[0]) if self._is_hybrid_collection(prefixed_name) else vectors[0]
        query_response = self.client.query_points(
            collection_name=prefixed_name,
            query=query,
            limit=limit,
            query_filter=query_filter,
        )
        get_result = self._result_to_get_result(query_response.points)
        return SearchResult(
            ids=get_result.ids,
            documents=get_result.documents,
            metadatas=get_result.metadatas,
            # qdrant distance is [-1, 1], normalize to [0, 1]
            distances=[[(point.score + 1.0) / 2.0 for point in query_response.points]],
        )

    def query(self, collection_name: str, filter: dict, limit: Optional[int] = None):
        # Construct the filter string for querying
        if not self.has_collection(collection_name):
            return None
        try:
            if limit is None:
                limit = NO_LIMIT  # otherwise qdrant would set limit to 10!

            field_conditions = []
            for key, value in filter.items():
                field_conditions.append(
                    models.FieldCondition(key=f'metadata.{key}', match=models.MatchValue(value=value))
                )

            points = self.client.scroll(
                collection_name=f'{self.collection_prefix}_{collection_name}',
                scroll_filter=models.Filter(should=field_conditions),
                limit=limit,
            )
            return self._result_to_get_result(points[0])
        except Exception as e:
            log.exception(f"Error querying a collection '{collection_name}': {e}")
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        # Get all the items in the collection.
        points = self.client.scroll(
            collection_name=f'{self.collection_prefix}_{collection_name}',
            limit=NO_LIMIT,  # otherwise qdrant would set limit to 10!
        )
        return self._result_to_get_result(points[0])

    def insert(self, collection_name: str, items: list[VectorItem]):
        # Insert the items into the collection, if the collection does not exist, it will be created.
        self._create_collection_if_not_exists(collection_name, len(items[0]['vector']))
        points = self._create_points(items)
        self.client.upload_points(f'{self.collection_prefix}_{collection_name}', points)

    def upsert(self, collection_name: str, items: list[VectorItem]):
        # Update the items in the collection, if the items are not present, insert them. If the collection does not exist, it will be created.
        self._create_collection_if_not_exists(collection_name, len(items[0]['vector']))
        prefixed_name = f'{self.collection_prefix}_{collection_name}'
        points = self._create_points(items, collection_is_hybrid=self._is_hybrid_collection(prefixed_name))
        return self.client.upsert(prefixed_name, points)

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        # Delete by point ID: the point ID is the item's id (see _create_points).
        # Filtering on metadata.id silently misses points whose payload omits an
        # id (e.g. memories), leaving orphaned vectors behind.
        if ids:
            return self.client.delete(
                collection_name=f'{self.collection_prefix}_{collection_name}',
                points_selector=models.PointIdsList(points=ids),
            )

        field_conditions = []
        if filter:
            for key, value in filter.items():
                field_conditions.append(
                    models.FieldCondition(
                        key=f'metadata.{key}',
                        match=models.MatchValue(value=value),
                    )
                )

        return self.client.delete(
            collection_name=f'{self.collection_prefix}_{collection_name}',
            points_selector=models.FilterSelector(filter=models.Filter(must=field_conditions)),
        )

    def reset(self):
        # Resets the database. This will delete all collections and item entries.
        collection_names = self.client.get_collections().collections
        for collection_name in collection_names:
            if collection_name.name.startswith(self.collection_prefix):
                self.client.delete_collection(collection_name=collection_name.name)


def _qdrant_hybrid_search(
    self: QdrantClient,
    collection_name: str,
    query: str,
    vectors: list[list[float | int]],
    filter: Optional[dict] = None,
    limit: int = 10,
    hybrid_bm25_weight: float = 0.5,
) -> Optional[SearchResult]:
    """
    Backend-native hybrid search: fuse dense (semantic) and sparse (BM25)
    retrieval for a single query, mirroring the pgvector contract.

    Bound onto ``QdrantClient.hybrid_search`` only when the optional
    ``fastembed`` dependency is importable (see module bottom), so the standard
    ``supports_hybrid_search`` detection advertises native hybrid support only
    when it can actually run. Returns None when the feature flag is off or the
    collection is not a hybrid collection, so the caller falls back to the
    legacy hybrid path.
    """
    if not self.client or not query or not query.strip():
        return None
    if self._get_sparse_encoder() is None:
        return None
    if limit is None:
        limit = NO_LIMIT
    limit = max(1, limit)

    prefixed_name = f'{self.collection_prefix}_{collection_name}'
    if not self.client.collection_exists(collection_name=prefixed_name):
        return None
    if not self._is_hybrid_collection(prefixed_name):
        return None

    bm25_weight = min(max(hybrid_bm25_weight, 0.0), 1.0)
    vector_weight = 1.0 - bm25_weight

    vector_result = None
    if vector_weight > 0 and vectors:
        vector_result = self.search(
            collection_name=collection_name,
            vectors=vectors,
            filter=filter,
            limit=limit,
        )

    fts_results: list[dict[str, Any]] = []
    if bm25_weight > 0:
        try:
            sparse_query = self._encode_sparse_query(query)
            sparse_filter = (
                models.Filter(must=[_metadata_filter(key, op, value) for key, op, value in iter_filter_conditions(filter)])
                if filter
                else None
            )
            sparse_response = self.client.query_points(
                collection_name=prefixed_name,
                query=sparse_query,
                using=self.QDRANT_SPARSE_VECTOR_NAME,
                limit=limit,
                query_filter=sparse_filter,
            )
            fts_results = [
                {
                    'id': point.id,
                    'text': point.payload.get('text', ''),
                    'vmetadata': point.payload.get('metadata', {}),
                }
                for point in sparse_response.points
            ]
        except Exception as e:
            log.exception('Error during Qdrant sparse (BM25) search: %s', e)
            return None

    return merge_hybrid_search_results(
        vector_result=vector_result,
        fts_results=fts_results,
        num_queries=1,
        limit=limit,
        hybrid_bm25_weight=hybrid_bm25_weight,
    )


if FASTEMBED_AVAILABLE:
    # Expose native hybrid search (and thereby advertise it via
    # supports_hybrid_search) only when the optional fastembed dependency is
    # present. Without it the class keeps the base no-op, so the pipeline uses
    # the legacy hybrid path.
    QdrantClient.hybrid_search = _qdrant_hybrid_search
