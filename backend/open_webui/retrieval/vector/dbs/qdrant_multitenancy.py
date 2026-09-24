"""
NOTE: This vector database integration is community-supported and maintained on a best-effort basis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import grpc
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
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import PointStruct, SparseVector
from qdrant_client.models import models

try:
    from fastembed.sparse import SparseTextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

NO_LIMIT = 999999999
TENANT_ID_FIELD = 'tenant_id'
DEFAULT_DIMENSION = 384

log = logging.getLogger(__name__)


def _tenant_filter(tenant_id: str) -> models.FieldCondition:
    return models.FieldCondition(key=TENANT_ID_FIELD, match=models.MatchValue(value=tenant_id))


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
            raise ValueError('QDRANT_URI is not set. Please configure it in the environment variables.')

        # Unified handling for either scheme
        parsed = urlparse(self.QDRANT_URI)
        host = parsed.hostname or self.QDRANT_URI
        http_port = parsed.port or 6333  # default REST port

        self.client = (
            Qclient(
                host=host,
                port=http_port,
                grpc_port=self.GRPC_PORT,
                prefer_grpc=self.PREFER_GRPC,
                api_key=self.QDRANT_API_KEY,
                timeout=self.QDRANT_TIMEOUT,
            )
            if self.PREFER_GRPC
            else Qclient(
                url=self.QDRANT_URI,
                api_key=self.QDRANT_API_KEY,
                timeout=self.QDRANT_TIMEOUT,
            )
        )

        # Main collection types for multi-tenancy
        self.MEMORY_COLLECTION = f'{self.collection_prefix}_memories'
        self.KNOWLEDGE_COLLECTION = f'{self.collection_prefix}_knowledge'
        self.FILE_COLLECTION = f'{self.collection_prefix}_files'
        self.WEB_SEARCH_COLLECTION = f'{self.collection_prefix}_web-search'
        self.HASH_BASED_COLLECTION = f'{self.collection_prefix}_hash-based'

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

    def _is_hybrid_collection(self, mt_collection_name: str) -> bool:
        """True if the collection stores sparse vectors (was created in hybrid mode)."""
        try:
            info = self.client.get_collection(mt_collection_name)
            return bool(info.config.params.sparse_vectors)
        except Exception:
            return False

    def _encode_sparse(self, texts: List[str]) -> List[SparseVector]:
        """Encode texts into sparse vectors using the fastembed sparse model."""
        embeddings = list(self._get_sparse_encoder().embed(texts))
        return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embeddings]

    def _encode_sparse_query(self, text: str) -> SparseVector:
        """Encode a query text into a sparse vector."""
        embeddings = list(self._get_sparse_encoder().query_embed(text))
        e = embeddings[0]
        return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

    def _build_hybrid_filter(self, tenant_id: str, filter: Optional[Dict]) -> models.Filter:
        """Tenant-scoped filter plus any metadata conditions."""
        conditions = [_tenant_filter(tenant_id)]
        if filter:
            conditions.extend(_metadata_filter(key, op, value) for key, op, value in iter_filter_conditions(filter))
        return models.Filter(must=conditions)

    def _result_to_get_result(self, points) -> GetResult:
        ids, documents, metadatas = [], [], []
        for point in points:
            payload = point.payload
            ids.append(point.id)
            documents.append(payload['text'])
            metadatas.append(payload['metadata'])
        return GetResult(ids=[ids], documents=[documents], metadatas=[metadatas])

    def _get_collection_and_tenant_id(self, collection_name: str) -> Tuple[str, str]:
        """
        Maps the traditional collection name to multi-tenant collection and tenant ID.

        Returns:
            tuple: (collection_name, tenant_id)

        WARNING: This mapping relies on current Open WebUI naming conventions for
        collection names. If Open WebUI changes how it generates collection names
        (e.g., "user-memory-" prefix, "file-" prefix, web search patterns, or hash
        formats), this mapping will break and route data to incorrect collections.
        POTENTIALLY CAUSING HUGE DATA CORRUPTION, DATA CONSISTENCY ISSUES AND INCORRECT
        DATA MAPPING INSIDE THE DATABASE.
        """
        # Check for user memory collections
        tenant_id = collection_name

        if collection_name.startswith('user-memory-'):
            return self.MEMORY_COLLECTION, tenant_id

        # Check for file collections
        elif collection_name.startswith('file-'):
            return self.FILE_COLLECTION, tenant_id

        # Check for web search collections
        elif collection_name.startswith('web-search-'):
            return self.WEB_SEARCH_COLLECTION, tenant_id

        # Handle hash-based collections (YouTube and web URLs)
        elif len(collection_name) == 63 and all(c in '0123456789abcdef' for c in collection_name):
            return self.HASH_BASED_COLLECTION, tenant_id

        else:
            return self.KNOWLEDGE_COLLECTION, tenant_id

    def _create_multi_tenant_collection(self, mt_collection_name: str, dimension: int = DEFAULT_DIMENSION):
        """
        Creates a collection with multi-tenancy configuration and payload indexes for tenant_id and metadata fields.

        When hybrid search is enabled, the collection is created with a named
        dense vector plus a sparse (BM25) vector so that `hybrid_search` can
        fuse the two signals.
        """
        if self.QDRANT_HYBRID_SEARCH_ENABLED:
            self.client.create_collection(
                collection_name=mt_collection_name,
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
                # Disable global index building due to multitenancy
                # For more details https://qdrant.tech/documentation/guides/multiple-partitions/#calibrate-performance
                hnsw_config=models.HnswConfigDiff(
                    payload_m=self.QDRANT_HNSW_M,
                    m=0,
                ),
            )
            log.info(
                'Multi-tenant hybrid collection %s created with dense (%s dims) + sparse vectors',
                mt_collection_name,
                dimension,
            )
        else:
            self.client.create_collection(
                collection_name=mt_collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    on_disk=self.QDRANT_ON_DISK,
                ),
                # Disable global index building due to multitenancy
                # For more details https://qdrant.tech/documentation/guides/multiple-partitions/#calibrate-performance
                hnsw_config=models.HnswConfigDiff(
                    payload_m=self.QDRANT_HNSW_M,
                    m=0,
                ),
            )
            log.info('Multi-tenant collection %s created with dimension %s!', mt_collection_name, dimension)

        self.client.create_payload_index(
            collection_name=mt_collection_name,
            field_name=TENANT_ID_FIELD,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
                on_disk=self.QDRANT_ON_DISK,
            ),
        )

        for field in ('metadata.hash', 'metadata.file_id'):
            self.client.create_payload_index(
                collection_name=mt_collection_name,
                field_name=field,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    on_disk=self.QDRANT_ON_DISK,
                ),
            )

    def _create_points(self, items: List[VectorItem], tenant_id: str, collection_is_hybrid: bool = False) -> List[PointStruct]:
        """
        Create point structs from vector items with tenant ID.

        When `collection_is_hybrid` is True the dense vector is stored under the
        named dense key and, if the sparse encoder is available, a sparse (BM25)
        vector is stored alongside it.
        """
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
                    payload={
                        'text': item['text'],
                        'metadata': process_metadata(item['metadata']),
                        TENANT_ID_FIELD: tenant_id,
                    },
                )
            )
        return points

    def _ensure_collection(self, mt_collection_name: str, dimension: int = DEFAULT_DIMENSION):
        """
        Ensure the collection exists and payload indexes are created for tenant_id and metadata fields.
        """
        if not self.client.collection_exists(collection_name=mt_collection_name):
            self._create_multi_tenant_collection(mt_collection_name, dimension)

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a logical collection exists by checking for any points with the tenant ID.
        """
        if not self.client:
            return False
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            return False
        tenant_filter = _tenant_filter(tenant_id)
        count_result = self.client.count(
            collection_name=mt_collection,
            count_filter=models.Filter(must=[tenant_filter]),
        )
        return count_result.count > 0

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Delete vectors by ID or filter from a collection with tenant isolation.
        """
        if not self.client:
            return None

        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, nothing to delete", mt_collection)
            return None

        must_conditions = [_tenant_filter(tenant_id)]
        if ids:
            # Delete by point ID within the tenant. The point ID is the item's id
            # (see _create_points); filtering on metadata.id silently misses points
            # whose payload omits an id (e.g. memories), leaving orphaned vectors.
            must_conditions.append(models.HasIdCondition(has_id=ids))
        elif filter:
            must_conditions += [_metadata_filter(k, '$eq', v) for k, v in filter.items()]

        return self.client.delete(
            collection_name=mt_collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=must_conditions)),
        )

    def search(
        self,
        collection_name: str,
        vectors: List[List[float | int]],
        filter: Optional[Dict] = None,
        limit: int = 10,
    ) -> Optional[SearchResult]:
        """
        Search for the nearest neighbor items based on the vectors with tenant isolation.
        """
        if not self.client or not vectors:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, search returns None", mt_collection)
            return None

        conditions = [_tenant_filter(tenant_id)]
        if filter:
            conditions.extend(_metadata_filter(key, op, value) for key, op, value in iter_filter_conditions(filter))
        # Hybrid collections store the dense vector under a named key, so the
        # query must address it explicitly.
        query = (self.QDRANT_DENSE_VECTOR_NAME, vectors[0]) if self._is_hybrid_collection(mt_collection) else vectors[0]
        query_response = self.client.query_points(
            collection_name=mt_collection,
            query=query,
            limit=limit,
            query_filter=models.Filter(must=conditions),
        )
        get_result = self._result_to_get_result(query_response.points)
        return SearchResult(
            ids=get_result.ids,
            documents=get_result.documents,
            metadatas=get_result.metadatas,
            distances=[[(point.score + 1.0) / 2.0 for point in query_response.points]],
        )

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        vectors: List[List[float | int]],
        filter: Optional[Dict] = None,
        limit: int = 10,
        hybrid_bm25_weight: float = 0.5,
    ) -> Optional[SearchResult]:
        """
        Backend-native hybrid search: fuse dense (semantic) and sparse (BM25)
        retrieval for a single query, mirroring the pgvector contract.

        Returns None when hybrid search is unavailable (disabled, `fastembed`
        missing, or the collection is not a hybrid collection) so the caller
        falls back to the legacy hybrid path.
        """
        if not self.client or not query or not query.strip():
            return None
        if self._get_sparse_encoder() is None:
            return None
        if limit is None:
            limit = NO_LIMIT
        limit = max(1, limit)

        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            return None
        if not self._is_hybrid_collection(mt_collection):
            return None

        bm25_weight = min(max(hybrid_bm25_weight, 0.0), 1.0)
        vector_weight = 1.0 - bm25_weight
        combined_filter = self._build_hybrid_filter(tenant_id, filter)

        vector_result = None
        if vector_weight > 0 and vectors:
            vector_result = self.search(
                collection_name=collection_name,
                vectors=vectors,
                filter=filter,
                limit=limit,
            )

        fts_results: List[Dict[str, Any]] = []
        if bm25_weight > 0:
            try:
                sparse_query = self._encode_sparse_query(query)
                sparse_response = self.client.query_points(
                    collection_name=mt_collection,
                    query=sparse_query,
                    using=self.QDRANT_SPARSE_VECTOR_NAME,
                    limit=limit,
                    query_filter=combined_filter,
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

    def query(self, collection_name: str, filter: Dict[str, Any], limit: Optional[int] = None):
        """
        Query points with filters and tenant isolation.
        """
        if not self.client:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, query returns None", mt_collection)
            return None
        if limit is None:
            limit = NO_LIMIT
        tenant_filter = _tenant_filter(tenant_id)
        field_conditions = [_metadata_filter(k, '$eq', v) for k, v in filter.items()]
        combined_filter = models.Filter(must=[tenant_filter, *field_conditions])
        points = self.client.scroll(
            collection_name=mt_collection,
            scroll_filter=combined_filter,
            limit=limit,
        )
        return self._result_to_get_result(points[0])

    def get(self, collection_name: str) -> Optional[GetResult]:
        """
        Get all items in a collection with tenant isolation.
        """
        if not self.client:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, get returns None", mt_collection)
            return None
        tenant_filter = _tenant_filter(tenant_id)
        points = self.client.scroll(
            collection_name=mt_collection,
            scroll_filter=models.Filter(must=[tenant_filter]),
            limit=NO_LIMIT,
        )
        return self._result_to_get_result(points[0])

    def upsert(self, collection_name: str, items: List[VectorItem]):
        """
        Upsert items with tenant ID.
        """
        if not self.client or not items:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        dimension = len(items[0]['vector'])
        self._ensure_collection(mt_collection, dimension)
        points = self._create_points(items, tenant_id, collection_is_hybrid=self._is_hybrid_collection(mt_collection))
        self.client.upload_points(mt_collection, points)
        return None

    def insert(self, collection_name: str, items: List[VectorItem]):
        """
        Insert items with tenant ID.
        """
        return self.upsert(collection_name, items)

    def reset(self):
        """
        Reset the database by deleting all collections.
        """
        if not self.client:
            return None
        for collection in self.client.get_collections().collections:
            if collection.name.startswith(self.collection_prefix):
                self.client.delete_collection(collection_name=collection.name)

    def delete_collection(self, collection_name: str):
        """
        Delete a collection.
        """
        if not self.client:
            return None
        mt_collection, tenant_id = self._get_collection_and_tenant_id(collection_name)
        if not self.client.collection_exists(collection_name=mt_collection):
            log.debug("Collection %s doesn't exist, nothing to delete", mt_collection)
            return None
        self.client.delete(
            collection_name=mt_collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[_tenant_filter(tenant_id)])),
        )
