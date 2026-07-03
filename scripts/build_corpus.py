#!/usr/bin/env python3
"""Build the retrieval corpus from pinned Arabic Wikipedia revisions.

Reads ``data/corpus/manifest.yaml``, fetches each article's plaintext extract
from the MediaWiki API, verifies the live revision still matches the pinned
``revid`` (failing loudly otherwise), chunks the text deterministically, and
writes the corpus plus a build-metadata sidecar.

Usage:
    python scripts/build_corpus.py \
        --manifest data/corpus/manifest.yaml \
        --out data/corpus/arwiki.jsonl

Network access to ``ar.wikipedia.org`` is required. The committed
``arwiki.jsonl`` is the frozen artifact; this script documents its provenance and
regenerates it when revisions match.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

from mizan.corpus_build import build_documents

API_URL = "https://ar.wikipedia.org/w/api.php"
USER_AGENT = "mizan-evals/0.1 (Arabic RAG eval corpus; contact via repo)"
#: Polite delay between requests, and backoff schedule for HTTP 429.
REQUEST_DELAY_S = 1.0
BACKOFF_S = (5.0, 15.0, 30.0)


class BuildError(RuntimeError):
    """Raised when the corpus cannot be built from the pinned manifest."""


def _fetch_extract(client: httpx.Client, title: str) -> tuple[str, int, str]:
    """Return ``(resolved_title, lastrevid, extract)`` for an article title.

    Retries with backoff on HTTP 429 (Wikipedia rate limiting).
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "extracts|info",
        "explaintext": "1",
        "exsectionformat": "plain",
        "titles": title,
    }
    for wait in (*BACKOFF_S, None):
        response = client.get(API_URL, params=params)
        if response.status_code == 429 and wait is not None:
            print(f"  rate limited on {title!r}; retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    pages = response.json()["query"]["pages"]
    if not pages or pages[0].get("missing"):
        raise BuildError(f"article not found: {title!r}")
    page = pages[0]
    return page["title"], page["lastrevid"], page.get("extract", "")


def build_corpus(manifest_path: Path, out_path: Path) -> dict:
    """Build the corpus JSONL and return build metadata."""
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    articles = manifest.get("articles") or []
    if not articles:
        raise BuildError(f"manifest has no articles: {manifest_path}")

    documents = []
    per_article: list[dict] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
        for index, entry in enumerate(articles):
            if index:
                time.sleep(REQUEST_DELAY_S)
            title, pinned = entry["title"], int(entry["revid"])
            resolved, lastrevid, extract = _fetch_extract(client, title)
            if resolved != title:
                raise BuildError(f"{title!r} redirected to {resolved!r}; pin the canonical title")
            if lastrevid != pinned:
                raise BuildError(
                    f"{title!r} advanced from pinned revid {pinned} to {lastrevid}; "
                    "review the article and re-pin the manifest before rebuilding"
                )
            docs = build_documents(title, pinned, extract)
            documents.extend(docs)
            per_article.append({"title": title, "revid": pinned, "chunks": len(docs)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")

    return {
        "source": manifest.get("source", "Arabic Wikipedia"),
        "license": manifest.get("license", "CC BY-SA 4.0"),
        "fetched_at": dt.datetime.now(dt.UTC).date().isoformat(),
        "api": API_URL,
        "n_articles": len(per_article),
        "n_documents": len(documents),
        "articles": per_article,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/manifest.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/corpus/arwiki.jsonl"))
    args = parser.parse_args(argv)

    try:
        meta = build_corpus(args.manifest, args.out)
    except (BuildError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    meta_path = args.out.with_name("BUILD_META.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"built {meta['n_documents']} documents from {meta['n_articles']} articles "
        f"-> {args.out} (meta: {meta_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
