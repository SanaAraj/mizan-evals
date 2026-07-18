# Retrieval corpus - Arabic Wikipedia extract

`arwiki.jsonl` is the document corpus used by the retrieval evaluation. It is
built from a fixed set of **pinned Arabic Wikipedia revisions** so that the
committed corpus is reproducible from a known source state.

## Provenance and licensing

- **Source:** [Arabic Wikipedia](https://ar.wikipedia.org).
- **License:** text is licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each document
  records the exact revision it was extracted from in its `url` field
  (`…/index.php?oldid=<revid>`); the pinned revisions are listed in
  `manifest.yaml` and mirrored in `BUILD_META.json`.
- Redistribution of this extract remains under CC BY-SA 4.0. This licence
  applies to the corpus text only, not to the surrounding evaluation code
  (MIT).

## How it is built

1. `manifest.yaml` pins each article title to an exact `revid`.
2. `scripts/build_corpus.py` fetches each article's plaintext extract from the
   MediaWiki API, and **refuses to build if the live revision no longer matches
   the pinned `revid`** - so a rebuild either reproduces the same source text or
   fails loudly asking you to review and re-pin.
3. Text is split into paragraphs and packed into deterministic chunks
   (`src/mizan/corpus_build.py`); ids follow `arwiki:<title-slug>#<chunk-index>`.

## Rebuild

```bash
PYTHONPATH=src python scripts/build_corpus.py \
  --manifest data/corpus/manifest.yaml \
  --out data/corpus/arwiki.jsonl
```

`BUILD_META.json` records the fetch date, per-article revision ids, and chunk
counts. The committed `arwiki.jsonl` is the frozen artifact; the script and
manifest document its provenance.
