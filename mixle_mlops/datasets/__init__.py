"""Build dataset generation — mixle's home turf.

A generative library makes *labeled* training/eval data with **verifiable labels**: sample from a fitted
mixle generative model (its sampler defines the exact data-generating process, so the labels are ground
truth by construction), or drive an LLM to emit structured JSON records against a schema.

* :mod:`generate` — :func:`generate_from_mixle`, :func:`generate_from_llm`, and the unified
  :func:`generate_dataset` dispatching on ``source`` and pulling the model from the registry.
* :mod:`export` — :func:`to_csv` / :func:`to_jsonl` / :func:`to_parquet` materialising a sampled dataset
  into the pluggable :class:`~mixle_mlops.multimodal.store.BlobStore`.
* :mod:`models` — the :class:`DatasetArtifact` table recording each generated dataset.
* :mod:`code_tasks` — the verified data factory for *code-writing* tasks (train a model to parse HTML by
  writing programs): pages rendered from known records (labels by construction), a subprocess sandbox,
  and execution-rejection-sampled ``(page -> code)`` + repair trajectories ready for ``/v1/fine_tunes``.
"""

from __future__ import annotations

from .code_tasks import (
    CodeTask,
    LLMTeacher,
    ReferenceTeacher,
    Trajectory,
    VerifiedDataset,
    evaluate,
    extract_code,
    harvest,
    make_task,
    render_page,
    run_parser,
    verify,
)
from .export import to_csv, to_jsonl, to_parquet
from .generate import (
    DatasetSpec,
    GeneratedDataset,
    generate_dataset,
    generate_from_llm,
    generate_from_mixle,
)

__all__ = [
    "DatasetSpec",
    "GeneratedDataset",
    "generate_dataset",
    "generate_from_llm",
    "generate_from_mixle",
    "to_csv",
    "to_jsonl",
    "to_parquet",
    "CodeTask",
    "Trajectory",
    "VerifiedDataset",
    "ReferenceTeacher",
    "LLMTeacher",
    "extract_code",
    "make_task",
    "render_page",
    "run_parser",
    "verify",
    "harvest",
    "evaluate",
]
