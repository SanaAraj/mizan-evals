# Retrieval evaluation set (arwiki-v1)

`eval_items.jsonl` is the retrieval slice of the mizan-evals harness: 15
information-need items, each expressed in parallel across **English**, **Modern
Standard Arabic (`msa`)**, and **Gulf dialect (`gulf`)**, with a single
language-independent gold label (`gold.relevant_doc_ids`).

- **Corpus:** `../corpus/arwiki.jsonl` - 12 pinned Arabic Wikipedia revisions
  (CC BY-SA 4.0). Gold ids are chunk ids in that corpus; every gold id resolves.
- **Construction:** each query targets a specific fact located in a specific
  corpus chunk; the gold chunk was read to confirm it answers the query.
- **Drafting and QA:** queries were LLM-drafted. An LLM QA pass (2026-07-03)
  checked cross-lingual semantic consistency (the en/msa/gulf variants ask the
  same question), MSA grammaticality, and Gulf dialect plausibility - six Gulf
  drafts that read as translated MSA were rephrased - and re-verified that
  every gold chunk contains the answer. Gold-label decisions (added
  alternates, one rejected candidate, the r-011 scoping) were approved by the
  maintainer via chat on 2026-07-03; each is recorded in that item's `notes`.
- **Review status:** every item carries `review_status: "llm_qa"`.
  **No native-speaker review has been completed yet**; that pass is pending,
  and items will move to `"native_reviewed"` per item as it lands.
- **Gold cardinality:** 13 items have a single gold chunk; `arwiki-r-002`
  (Saudi naming sequence) has three and `arwiki-r-004` (al-Qarawiyyin founding)
  has two, where multiple chunks genuinely answer the query. Recall is defined
  over the full gold set, so these items reward retrieving every relevant chunk.

## Split hygiene

This is an **evaluation** set. It shares no items with any training data. The
downstream fine-tuning project (see `briefs/03`) must construct its training
data from separate intents and document the separation explicitly; this file is
frozen while that project runs so the test is not tuned to the model.

## Scoring

`recall@k` and `MRR` are computed per (item, language) by the retriever under
test; see `src/mizan/scoring/retrieval.py`. For lexical (BM25) retrieval the
headline comparison is **msa vs gulf**: a lexical retriever over an MSA corpus
is expected to lose recall on dialectal phrasings, which is the signal this
slice is designed to expose. The **en** variants share almost no surface forms
with the Arabic corpus, so the en lexical baseline is near zero by
construction; it was measured once, archived under `runs/archive/`, and kept
out of the headline table. English remains fully in scope for the
tool-calling evaluation, where it is the project's headline comparison.
