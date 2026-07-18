# Sample evaluation items

These **10 items are a format reference only** - they exist to demonstrate the
schema and to give the loader something real to validate against. They are not
the evaluation set. The full dataset (100–200 items) is authored and
native-speaker reviewed separately.

Each line in `eval_items.jsonl` is one [`EvalItem`](../../src/mizan/schemas.py):
a single intent expressed in parallel across **English**, **Modern Standard
Arabic (`msa`)**, and **Gulf dialect (`gulf`)**, with a single language-
independent gold label.

## What the samples cover

| Task type        | Count | Gold signal                          |
|------------------|-------|--------------------------------------|
| `retrieval`      | 3     | `gold.relevant_doc_ids`              |
| `answer_quality` | 2     | per-variant `reference_answer`       |
| `faithfulness`   | 1     | `reference_answer` + context doc ids |
| `tool_calling`   | 4     | `gold.expected_tool`                 |

## Notes and known placeholders

- **Document ids are illustrative** (e.g. `arwiki:الرياض#0`). They follow a
  stable `arwiki:<title>#<chunk>` shape but do not yet resolve to a built
  corpus; the RAG corpus is a later milestone.
- **Tool arguments are stored in a single canonical (English) form.** Whether
  Arabic queries should expect Arabic-valued arguments (e.g. `"الرياض"` vs
  `"Riyadh"`) is an open methodology decision, not yet settled.
- Gulf-dialect phrasings (`وش`, `شلون`, `باچر`, `أدق`) are first drafts pending
  native-speaker review.
