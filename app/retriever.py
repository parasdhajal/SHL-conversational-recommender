"""
Hybrid retrieval: sentence-transformers embeddings + FAISS (cosine via inner product)
plus lightweight keyword scoring for Recall@10-style robustness.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

LOG = logging.getLogger("shl_agent")

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Query hints for personality / behavior hiring intents
_PERSONALITY_QUERY_RE = re.compile(
    r"\b(personality|behavioral|behaviour|communication|culture\s*fit|leadership|"
    r"workplace\s+behavior|workplace\s+behaviour|opq|traits?|temperament|interpersonal)\b",
    re.I,
)
# Down-rank documentation-style catalog rows during personality queries
_DOC_LIKE_NAME_RE = re.compile(
    r"\b(report|development report|framework|guide|interpretation|global skills)\b",
    re.I,
)
# Up-rank actual assessments / inventories
_ASSESSMENT_LIKE_RE = re.compile(
    r"\b(questionnaire|inventory|personality|behavioral|behaviour|opq|dependability|motivation)\b",
    re.I,
)


def _app_dir() -> Path:
    return Path(__file__).resolve().parent


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


@dataclass
class CatalogItem:
    name: str
    url: str
    test_type: str
    description: str
    raw: dict[str, Any]

    def embedding_text(self) -> str:
        return f"{self.name}\n{self.test_type}\n{self.description}"


@dataclass
class RetrievedItem:
    item: CatalogItem
    semantic_score: float
    keyword_score: float
    combined_score: float


class HybridRetriever:
    def __init__(
        self,
        catalog_path: Path | None = None,
        index_dir: Path | None = None,
        model_name: str | None = None,
        w_semantic: float = 0.82,
        w_keyword: float = 0.18,
        faiss_prefetch: int = 60,
    ) -> None:
        self.catalog_path = catalog_path or (_app_dir() / "catalog.json")
        self.index_dir = index_dir or (_app_dir() / "faiss_index")
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.w_semantic = w_semantic
        self.w_keyword = w_keyword
        self.faiss_prefetch = faiss_prefetch
        self._items: list[CatalogItem] = []
        self._doc_freq: dict[str, int] = {}
        self._doc_tokens: list[list[str]] = []
        self._mean_doc_len: float = 1.0
        self._model: SentenceTransformer | None = None
        self._index: Any | None = None
        self._vectors: np.ndarray | None = None

    def load(self) -> None:
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Missing catalog: {self.catalog_path}")
        raw_list = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self._items = []
        for row in raw_list:
            self._items.append(
                CatalogItem(
                    name=str(row.get("name", "")).strip(),
                    url=str(row.get("url", "")).strip(),
                    test_type=str(row.get("test_type", "")).strip(),
                    description=str(row.get("description", "")).strip(),
                    raw=row,
                )
            )
        self._doc_tokens = [tokenize(it.embedding_text()) for it in self._items]
        df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._doc_freq = df
        lengths = [len(x) for x in self._doc_tokens]
        self._mean_doc_len = float(np.mean(lengths)) if lengths else 1.0
        index_file = self.index_dir / "index.faiss"
        meta_file = self.index_dir / "meta.json"
        vecs_file = self.index_dir / "vectors.npy"
        if not index_file.is_file() or not meta_file.is_file() or not vecs_file.is_file():
            raise FileNotFoundError(
                f"FAISS index missing under {self.index_dir}. Run: python -m app.retriever --build"
            )
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("model_name") != self.model_name:
            LOG.warning("Index model %r != runtime %r; rebuild index.", meta.get("model_name"), self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._index = faiss.read_index(str(index_file))
        self._vectors = np.load(vecs_file)
        LOG.info("Loaded %d catalog items and FAISS index", len(self._items))

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def keyword_scores(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        n_docs = len(self._doc_tokens)
        if not q_tokens or n_docs == 0:
            return np.zeros(n_docs, dtype=np.float32)
        scores = np.zeros(n_docs, dtype=np.float32)
        for qi, t in enumerate(q_tokens):
            idf = math.log((1.0 + n_docs) / (1.0 + self._doc_freq.get(t, 0))) + 1.0
            for di, doc_toks in enumerate(self._doc_tokens):
                tf = doc_toks.count(t)
                if tf:
                    norm_len = len(doc_toks) / max(self._mean_doc_len, 1.0)
                    scores[di] += (1.0 + math.log(tf)) * idf * (1.2 / (1.0 + 0.75 * norm_len))
        if float(scores.max()) <= 0:
            return scores
        return scores / float(scores.max())

    def category_adjustment_scores(self, query: str) -> np.ndarray:
        """
        Soft boosts/penalties when the query signals personality/behavior hiring.
        Improves ranking without hardcoding final recommendations.
        """
        n_docs = len(self._items)
        adj = np.zeros(n_docs, dtype=np.float32)
        if not _PERSONALITY_QUERY_RE.search(query):
            return adj
        for di, item in enumerate(self._items):
            name = item.name
            blob = f"{name} {item.description} {item.test_type}"
            # SHL type P = Personality & Behavior
            if re.search(r"(^|/)P(/|$)", item.test_type):
                adj[di] += 0.14
            if _ASSESSMENT_LIKE_RE.search(blob):
                adj[di] += 0.10
            if "questionnaire" in name.lower():
                adj[di] += 0.12
            if _DOC_LIKE_NAME_RE.search(name):
                adj[di] -= 0.22
            # Reports derived from OPQ are valid but rank below core questionnaires
            if name.lower().startswith("opq ") and "report" in name.lower():
                adj[di] -= 0.06
        mx = float(adj.max())
        mn = float(adj.min())
        if mx > mn:
            adj = (adj - mn) / (mx - mn)
        return adj

    def semantic_scores_prefetch(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices [k], semantic_scores_normalized_to_[0,1] for those indices)."""
        if self._index is None or self._vectors is None:
            raise RuntimeError("Retriever not loaded")
        model = self._ensure_model()
        qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims, idxs = self._index.search(qv.astype(np.float32), min(self.faiss_prefetch, len(self._items)))
        sim_row = sims[0]
        idx_row = idxs[0]
        mask = idx_row >= 0
        idx_row = idx_row[mask]
        sim_row = sim_row[mask]
        if sim_row.size == 0:
            return idx_row.astype(np.int64), np.zeros_like(sim_row, dtype=np.float32)
        smin, smax = float(sim_row.min()), float(sim_row.max())
        span = smax - smin if smax > smin else 1.0
        norm = (sim_row - smin) / span
        return idx_row.astype(np.int64), norm.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        if not self._items:
            raise RuntimeError("Empty catalog")
        top_k = max(1, min(10, top_k))
        idx_pref, sem_norm = self.semantic_scores_prefetch(query)
        kw_all = self.keyword_scores(query)
        cat_all = self.category_adjustment_scores(query)
        combined: list[RetrievedItem] = []
        for j, di in enumerate(idx_pref.tolist()):
            sem = float(sem_norm[j]) if j < len(sem_norm) else 0.0
            kw = float(kw_all[di]) if di < len(kw_all) else 0.0
            cat = float(cat_all[di]) if di < len(cat_all) else 0.0
            comb = self.w_semantic * sem + self.w_keyword * kw + 0.12 * cat
            combined.append(
                RetrievedItem(
                    item=self._items[di],
                    semantic_score=sem,
                    keyword_score=kw,
                    combined_score=comb,
                )
            )
        combined.sort(key=lambda r: r.combined_score, reverse=True)
        return combined[:top_k]


def build_faiss_index(
    catalog_path: Path | None = None,
    index_dir: Path | None = None,
    model_name: str | None = None,
) -> None:
    catalog_path = catalog_path or (_app_dir() / "catalog.json")
    index_dir = index_dir or (_app_dir() / "faiss_index")
    model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    items: list[CatalogItem] = []
    for row in json.loads(catalog_path.read_text(encoding="utf-8")):
        items.append(
            CatalogItem(
                name=str(row.get("name", "")).strip(),
                url=str(row.get("url", "")).strip(),
                test_type=str(row.get("test_type", "")).strip(),
                description=str(row.get("description", "")).strip(),
                raw=row,
            )
        )
    texts = [it.embedding_text() for it in items]
    LOG.info("Encoding %d catalog rows with %s", len(texts), model_name)
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    vecs = _l2_normalize(np.asarray(vecs, dtype=np.float32))
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    np.save(index_dir / "vectors.npy", vecs)
    (index_dir / "meta.json").write_text(
        json.dumps({"model_name": model_name, "dim": dim, "count": len(items)}, indent=2),
        encoding="utf-8",
    )
    LOG.info("Wrote FAISS index to %s", index_dir)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Build FAISS index for catalog.json")
    p.add_argument("--build", action="store_true", help="Build index from catalog")
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    if args.build:
        build_faiss_index(catalog_path=args.catalog, index_dir=args.out_dir, model_name=args.model)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
