"""MEIS Phase 2 — cross-domain prior library retrieval.

Dead-simple bag-of-words retrieval over a curated JSON prior library.
Deliberately primitive: Step 3 will tell us what structure the LLM
actually needs. Only upgrade (embeddings, typed queries) when that
signal arrives.

Usage:
    from phase2_prior_library.retrieval import PriorLibrary
    lib = PriorLibrary.load_default()
    hits = lib.retrieve("weight height cube", k=3)
    for h in hits:
        print(h["id"], "::", h["statement"][:80])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LIB_DIR = Path(__file__).parent
DEFAULT_FILES = [
    "human_body.json",
    "growth_curves.json",
]


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9_]+", text.lower()) if t}


@dataclass
class PriorLibrary:
    entries: list[dict[str, Any]]

    # -- Loading --
    @classmethod
    def load_default(cls) -> "PriorLibrary":
        all_entries: list[dict] = []
        for fname in DEFAULT_FILES:
            all_entries.extend(cls._load_file(_LIB_DIR / fname))
        return cls(entries=all_entries)

    @classmethod
    def load_files(cls, paths: list[str | Path]) -> "PriorLibrary":
        entries: list[dict] = []
        for p in paths:
            entries.extend(cls._load_file(Path(p)))
        return cls(entries=entries)

    @staticmethod
    def _load_file(path: Path) -> list[dict]:
        data = json.loads(path.read_text())
        assert isinstance(data, list), f"expected JSON list at {path}, got {type(data).__name__}"
        for e in data:
            assert "id" in e and "keywords" in e and "statement" in e, f"bad entry in {path}: missing required fields"
        return data

    # -- Retrieval --
    def retrieve(self, query: str, k: int = 5,
                 domain: str | None = None) -> list[dict[str, Any]]:
        """Return the top-k entries scored by keyword-token overlap with `query`.

        Tokens matched across: entry['keywords'] (primary) + entry['vars_involved']
        + tokens drawn from entry['statement'] (weight 0.3).
        If `domain` is provided, restricts search to entries where entry['domain'] == domain.
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for entry in self.entries:
            if domain is not None and entry.get("domain") != domain:
                continue

            kw_tokens = {t for kw in entry.get("keywords", []) for t in _tokenize(kw)}
            var_tokens = {t for v in entry.get("vars_involved", []) for t in _tokenize(v)}
            stmt_tokens = _tokenize(entry.get("statement", ""))

            score = (
                3.0 * len(q_tokens & kw_tokens)
                + 2.0 * len(q_tokens & var_tokens)
                + 0.3 * len(q_tokens & stmt_tokens)
            )
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    # -- Formatting --
    @staticmethod
    def format_for_prompt(entries: list[dict], max_chars: int = 2000) -> str:
        """Render retrieved entries as a compact system-prompt chunk.

        Each entry: id (domain) — statement. If a `formal` block exists, append a
        one-line parametric summary.
        """
        parts: list[str] = []
        for e in entries:
            line = f"- [{e['id']}] ({e.get('domain','?')}): {e['statement']}"
            formal = e.get("formal")
            if isinstance(formal, dict):
                if "relation" in formal:
                    line += f"\n    Formal: {formal['relation']}"
                if "parameters" in formal:
                    line += f"\n    Params: {formal['parameters']}"
                elif "distribution" in formal:
                    line += f"\n    Distribution: {formal['distribution']}"
            parts.append(line)
        blob = "\n".join(parts)
        return blob[:max_chars]


if __name__ == "__main__":
    lib = PriorLibrary.load_default()
    print(f"Loaded {len(lib.entries)} entries from default files.\n")
    for query in [
        "predict weight given height",
        "density body mass",
        "shoe size foot",
        "footprint depth sand",
        "pressure force area",
    ]:
        print(f">>> query: {query!r}")
        hits = lib.retrieve(query, k=3)
        for h in hits:
            print(f"   {h['id']:35s} domain={h.get('domain'):20s}")
        print()
