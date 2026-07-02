"""mizan-evals: evaluation harness for Arabic RAG and agent tool-calling.

The package is organised around a few small, independent pieces:

- :mod:`mizan.schemas` - the parallel English / MSA / Gulf-dialect evaluation
  item format and its gold labels.
- :mod:`mizan.dataset` - loading and validating evaluation items from JSONL.
- :mod:`mizan.config` - the run configuration loaded from YAML.
- :mod:`mizan.results` - run metadata and per-item result records.
- :mod:`mizan.llm` - a disk-cached, resumable LLM client with a mock backend.
- :mod:`mizan.scoring` - retrieval metrics (recall@k, MRR).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
