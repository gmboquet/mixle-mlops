Release Notes
=============

``mixle-mlops`` hosts Mixle models and OpenAI-compatible LLMs behind one
gateway, with accounts and API keys, RAG and multimodal input, an MCP server,
feedback and self-evolution loops, dataset export, and deployment helpers for
local and cloud environments.

0.7.0
-----

0.7.0 is the first version of ``mixle-mlops`` published to PyPI (see
:doc:`installation`) and the first checklist-verified release: coordinated
with ``mixle`` core's 0.7.0, it went through a full release-readiness pass
(see ``release-checklists/0.7.0.md`` in the repository root) rather than only
a documentation-completion pass. That pass built the sdist/wheel and installed
them into a fresh, isolated environment; ran the full test suite (route,
provider, feedback, dataset, RAG, account, storage, MCP, multimodal, pool,
substrate, telemetry, creation-verb, and drift-retraining coverage) as one
release gate; and fixed several real bugs it found along the way, including a
``transformers>=5`` incompatibility in the local logit engine's KV-cache trie,
an undeclared base dependency on ``fsspec`` that broke the local object-store
backend outside the ``cloud`` extra, and a cloud deployment that could boot
silently on the default secret key. See the root ``CHANGELOG.md`` for the full
list.

0.7.0 also adds LoRA/QLoRA fine-tuning depth: SFT (chat-template + prompt
masking) alongside the existing raw-continuation training, `resume_from`
checkpoint continuation, and -- the piece that was previously missing --
adapter-aware serving (`HFLogitProvider`'s `adapter_path`) plus `POST
/v1/models/load` to bring a completed fine-tune live without a restart. See
:doc:`gateway-capabilities`'s LoRA/QLoRA section.

0.6.3
-----

0.6.3 is a documentation-completion release: the gateway, registry, training,
and deployment surfaces built in earlier commits are now documented well
enough to review. The manual covers the route-by-route operating contract
(:doc:`gateway-operating-contract`), the substrate serving, telemetry, pool
job, and creation-verb surfaces (:doc:`gateway-capabilities`), and
distillation and cross-modal training
(:doc:`model-distillation-and-cross-modal-training`), backed by a generated
API reference for every public module and route.

Known Risks
-----------

Resolved in 0.7.0's release-readiness pass:

* a clean package build and install from a fresh environment;
* route and service tests run together as a single release gate, not only
  per-surface (see :doc:`validation` for the current focused test commands).

Still open:

* no CI is configured in this repository, so every gate above was verified
  locally rather than by a reproducible, re-runnable pipeline;
* mock/skip/real status is recorded per test run, but not yet collected into
  one per-service table;
* verification that this package still co-installs cleanly with the rest of
  the Mixle family is tracked outside this repository, not by this manual;
* the example scripts under ``examples/`` are inventoried in
  :doc:`example-execution-manifest` but have not all been executed and
  classified for this release.
