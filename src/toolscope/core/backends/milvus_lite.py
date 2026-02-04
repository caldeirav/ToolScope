from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

from ..types import CanonicalTool, ToolIndexBackend, ToolFingerprint


class MilvusLiteBackend(ToolIndexBackend):
    """
    Embedded/local Milvus backend (Milvus Lite).

    Requires:
      pip install pymilvus

    Uses a local URI (file-based or local service depending on pymilvus version).
    """

    def __init__(
        self,
        *,
        uri: str = "toolscope_milvus.db",
        default_namespace: str = "__default__",
        metric_type: str = "IP",  # inner product / cosine
        index_type: str = "AUTOINDEX",
    ) -> None:
        self._uri = uri
        self._default_namespace = default_namespace
        self._metric_type = metric_type
        self._index_type = index_type
        self._connected = False
        self._dims: Dict[str, int] = {}

    # ---------- internal helpers ----------

    def _connect(self) -> None:
        if self._connected:
            return
        from pymilvus import connections
        connections.connect(alias="default", uri=self._uri)
        self._connected = True

    # ---------- ToolIndexBackend ----------

    def ensure_namespace(self, *, namespace: Optional[str], dim: int) -> None:
        """
        Ensure a Milvus collection exists for this namespace with the required schema.

        Schema (minimal but sufficient for round-tripping + filtering):
          - id (VARCHAR primary key): ToolFingerprint.value
          - vector (FLOAT_VECTOR): embedding
          - name, description (VARCHAR)
          - schema_json, tags_json, payload_json (VARCHAR JSON blobs)
        """
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                FieldSchema,
                DataType,
                utility,
            )
        except Exception as e:
            raise ImportError(
                "MilvusLiteBackend requires 'pymilvus'. Install with: pip install toolscope[milvus]") from e

        self._connect()
        ns = namespace or self._default_namespace

        if ns in self._dims:
            if self._dims[ns] != dim:
                raise ValueError(f"Milvus collection '{ns}' dim mismatch: expected {self._dims[ns]}, got {dim}.")
            return

        if utility.has_collection(ns):
            # Best-effort: assume dim is correct; to be strict, you can inspect schema and vector dim.
            self._dims[ns] = dim
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),

            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),

            # Keep these fairly small; descriptions can be big in reality, but embedding truncation
            # means you often don't need the full thing stored. Still, we store full description
            # up to a reasonable cap.
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),

            # JSON blobs (sizes are tradeoffs; tune if you have very large schemas/payloads)
            FieldSchema(name="schema_json", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="tags_json", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="payload_json", dtype=DataType.VARCHAR, max_length=16384),
        ]

        schema = CollectionSchema(fields=fields, description="ToolScope tool index")
        col = Collection(name=ns, schema=schema)

        # Vector index
        col.create_index(
            field_name="vector",
            index_params={"index_type": self._index_type, "metric_type": self._metric_type},
        )

        col.load()
        self._dims[ns] = dim

    def upsert(
            self,
            ids: Sequence[str],
            vectors: Sequence[List[float]],
            payloads: Sequence[CanonicalTool],
            *,
            namespace: Optional[str] = None,
    ) -> None:
        """
        Insert/update tool vectors + metadata.

        We implement upsert as:
          1) delete existing ids (best-effort)
          2) insert new rows
        """
        if not vectors:
            return

        ns = namespace or self._default_namespace
        dim = len(vectors[0])
        self.ensure_namespace(namespace=ns, dim=dim)

        self._connect()
        from pymilvus import Collection
        import json

        col = Collection(ns)

        # Validate dimensions
        for v in vectors:
            if len(v) != dim:
                raise ValueError(f"Vector dimension mismatch in upsert: expected {dim}, got {len(v)}")

        # Best-effort delete for same PKs (Milvus "upsert" support differs by version)
        try:
            expr = f'id in [{",".join(json.dumps(i) for i in ids)}]'
            col.delete(expr)
        except Exception:
            pass

        # Build columns
        col_ids: List[str] = []
        col_vecs: List[List[float]] = []
        col_name: List[str] = []
        col_desc: List[str] = []
        col_schema: List[str] = []
        col_tags: List[str] = []
        col_payload: List[str] = []

        for _id, vec, tool in zip(ids, vectors, payloads, strict=True):
            col_ids.append(_id)
            col_vecs.append(vec)
            col_name.append(tool.name or "")
            col_desc.append(tool.description or "")
            col_schema.append(json.dumps(tool.input_schema or {}, ensure_ascii=False))
            col_tags.append(json.dumps(list(tool.tags or ()), ensure_ascii=False))

            # payload should be JSON-serializable for persistence backends
            try:
                col_payload.append(json.dumps(tool.payload, ensure_ascii=False))
            except TypeError:
                # Fallback: store null if not serializable (round-trip won't work for this tool)
                col_payload.append("null")

        # PyMilvus insert supports "list of fields" in schema order
        rows = [
            col_ids,
            col_vecs,
            col_name,
            col_desc,
            col_schema,
            col_tags,
            col_payload,
        ]

        col.insert(rows)
        col.flush()

    def delete(self, ids: Sequence[str], *, namespace: Optional[str] = None) -> None:
        from pymilvus import Collection, utility

        self._connect()
        ns = namespace or self._default_namespace
        if not utility.has_collection(ns):
            return

        col = Collection(ns)
        expr = f'id in [{",".join(json.dumps(i) for i in ids)}]'
        col.delete(expr)
        col.flush()

    def search(
        self,
        query_vector: List[float],
        *,
        k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        from pymilvus import Collection, utility

        self._connect()
        ns = namespace or self._default_namespace
        if not utility.has_collection(ns):
            return []

        col = Collection(ns)
        col.load()

        res = col.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": self._metric_type},
            limit=k,
            output_fields=["id"],
        )

        hits: List[Tuple[str, float]] = []
        for h in res[0]:
            hits.append((str(h.id), float(h.distance)))
        return hits

    def get(self, ids: Sequence[str], *, namespace: Optional[str] = None) -> List[CanonicalTool]:
        from pymilvus import Collection, utility
        import json

        self._connect()
        ns = namespace or self._default_namespace
        if not utility.has_collection(ns) or not ids:
            return []

        col = Collection(ns)
        expr = f'id in [{",".join(json.dumps(i) for i in ids)}]'
        rows = col.query(
            expr,
            output_fields=["id", "name", "description", "schema_json", "tags_json", "payload_json"],
        )

        out: List[CanonicalTool] = []
        for r in rows:
            tool_id = str(r.get("id", ""))

            try:
                schema = json.loads(r.get("schema_json") or "{}")
            except Exception:
                schema = {}

            try:
                tags_list = json.loads(r.get("tags_json") or "[]")
                tags = tuple(t for t in tags_list if isinstance(t, str))
            except Exception:
                tags = ()

            payload = None
            try:
                payload = json.loads(r.get("payload_json") or "null")
            except Exception:
                payload = None

            out.append(
                CanonicalTool(
                    name=r.get("name") or "",
                    description=r.get("description") or "",
                    input_schema=schema,
                    tags=tags,
                    fingerprint=ToolFingerprint(tool_id),
                    payload=payload,
                )
            )
        return out

