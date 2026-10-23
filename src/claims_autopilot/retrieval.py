"""Small retrieval layer for RAG (retrieval-augmented generation).

The knowledge base is (a) every row of the CARC/RARC reference CSV and (b)
each section of data/denial_knowledge_base.md. We rank passages with BM25, a
classic keyword-ranking formula, implemented in ~30 lines so a beginner can
read it. No embeddings, vector database, or extra services are needed for a
corpus this small, and results are deterministic and testable.

Exact code lookups (e.g. "N265") never depend on ranking: `lookup_code`
reads the reference table directly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .config import KNOWLEDGE_PATH, REFERENCE_PATH

_STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were with not".split())


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str
    source: str


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@lru_cache(maxsize=1)
def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_PATH, dtype=str).fillna("")
    df["code"] = df["code"].str.strip().str.upper()
    return df


def lookup_code(code_type: str, code: str) -> dict | None:
    df = load_reference()
    hit = df[(df["code_type"] == code_type) & (df["code"] == code.upper())]
    return None if hit.empty else hit.iloc[0].to_dict()


@lru_cache(maxsize=1)
def load_passages() -> tuple[Passage, ...]:
    passages: list[Passage] = []
    for row in load_reference().itertuples():
        passages.append(Passage(
            id=f"{row.code_type}-{row.code}",
            title=f"{row.code_type} {row.code}",
            text=(f"{row.code_type} {row.code}: {row.official_description} "
                  f"Plain language: {row.plain_language_project_note} "
                  f"Suggested actions: {row.suggested_actions_project_guidance}"),
            source=row.source_url,
        ))
    md = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    for block in re.split(r"\n(?=## )", md):
        m = re.match(r"## (KB-[A-Z\-]+): (.+)\n", block)
        if m:
            passages.append(Passage(id=m.group(1), title=m.group(2), text=block[m.end():].strip(),
                                    source="data/denial_knowledge_base.md (project-written guidance)"))
    return tuple(passages)


class BM25:
    def __init__(self, passages: tuple[Passage, ...], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.docs = [_tokens(p.title + " " + p.text) for p in passages]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[tuple[Passage, float]]:
        q = _tokens(query)
        results = []
        for p, doc in zip(self.passages, self.docs):
            tf = Counter(doc)
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            if s > 0:
                results.append((p, s))
        return sorted(results, key=lambda x: -x[1])


@lru_cache(maxsize=1)
def _index() -> BM25:
    return BM25(load_passages())


def retrieve(query: str, k: int = 6, must_include: list[str] | None = None) -> list[Passage]:
    """Top-k passages for `query`. Passage ids in `must_include` (the exact
    codes found in the denial) are always returned first."""
    by_id = {p.id: p for p in load_passages()}
    out: list[Passage] = [by_id[i] for i in (must_include or []) if i in by_id]
    for p, _ in _index().score(query):
        if len(out) >= k + len(must_include or []):
            break
        if p not in out:
            out.append(p)
    return out
