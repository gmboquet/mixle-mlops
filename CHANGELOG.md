# Changelog

All notable changes to `mixle-mlops` are documented here. This is the first versioned,
checklist-verified release; earlier history (84+ commits building the platform from a single-model
scoring server into the full gateway, including an undocumented 0.6.3 documentation-completion pass)
predates formal changelog tracking.

## [0.7.0] — 2026-07-11

Coordinated with `mixle` core's 0.7.0 release. First release to go through a full
release-readiness pass (see `release-checklists/0.7.0.md`).

### Added

- OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/models`, streaming) hosting open/hosted
  LLMs (Ollama, vLLM, llama.cpp, hosted OpenAI-compatible or native Anthropic/Gemini backends) and
  fitted `mixle` models side by side (`/v1/mixle/{predict,score,latent,decide}`).
- The mixle bridge stack: best-of-N self-consistency, a FrugalGPT-style cascade router, Mixture-of-
  Agents, and program-offload (exact arithmetic/probability via the `mixle_solve` tool).
- Self-evolution: `POST /v1/evolve/{model}` and `/v1/evolve/tick` run `mixle.evolve`'s measure →
  propose → verify → promote loop against served models, with rollback and self-calibrating cascade
  thresholds.
- Accounts, API keys, and OAuth (Sign in with Google/Apple); multimodal image input; RAG over
  uploaded PDF/DOCX/PPTX; image generation (hosted or local Stable Diffusion via `diffusers`);
  dataset generation/export; persisted conversations with json/markdown/PDF export.
- Caching (memory/Redis) and rate limiting; an MCP server; a server-side agentic loop (tool calling,
  MCP tools, RAG, mixle decide/predict, exact compute).
- The pool route (`/v1/pool/*`): a job gateway that runs budget-reject and billable-confirm rails
  *before* any work executes, so an over-budget or unconfirmed-cost job is rejected at zero cost.
  Executes `fit` jobs server-side; `GET /jobs/{id}/artifact` round-trips the fitted artifact
  bit-exact (verified by reloading and matching the parameter fingerprint); `GET /pool/spend` is
  the per-user spend/quota ledger.
- The substrate route (`/v1/substrate/*`): the knowledge substrate and all-data RAG deployed over
  HTTP, with `/context` (budgeted, cited context packets, no answer generation) and `/factuality`
  (per-claim grounding: extraction, retrieval, content-overlap corroboration, naming any
  unsupported claim). Cross-team sharing is audited (`/publish`, scope-guarded) and promotion into
  a curated scope is gated (`/propose` → `/pending` → `/approve`|`/reject`, approval admin-only).
- The verbs route (`/v1/create|uq|simulate|synthesize|skills`): HTTP twins for the core creation
  verbs over a per-user model store — certified fit, an honest "not quantifiable" UQ result when a
  family can't be Laplace-flattened, synthetic draws with declarative constraints, and a findable
  skill registry. `GET /v1/lineage/{model_id}` answers what a model was fit from and what depends on
  it (data fingerprint, parameter fingerprint, certificate, dependent skills) as queryable edges.
- The telemetry route (`/v1/telemetry`): the shared sink typed, PII-free decision events (fit,
  placement, route, escalation, pool job, drift) are read back from as `(features, choice, outcome)`
  rows for learned orchestration.
- Served model manifests (`GET /v1/models/{id}`) now carry a certificate (guarantee ladder,
  why-not-ADAM audit) and calibration status when the fit reserved a holdout, honestly reporting
  `null` rather than fabricating either when the model can't support them.
- The automated drift-detect → retrain → register → promote loop (`mixle-drift-retrain`) is
  exercised end to end: a drift-tripping batch promotes a new production version and rolls the
  reference forward; an in-distribution batch leaves the production alias untouched.
- A local logit-level engine (token-level PoE + grammar-masked constrained decoding via
  `transformers`), including a KV-cache trie (`TreeLogitProvider`) for efficient tree-structured
  decoding/enumeration.
- Multi-cloud deploy: Docker/Helm/Terraform for AWS/Azure/GCP/Alicloud, a unified `fsspec`-backed
  object store, and generic GPU-compute launch (rented boxes via vast.ai, or any OpenAI-compatible
  endpoint).
- LoRA/QLoRA fine-tuning depth: the generated training script now supports supervised fine-tuning
  (a `messages`-column dataset applies the base model's chat template and masks the prompt out of
  the loss) alongside its original raw-continuation training, plus `resume_from` to continue an
  existing adapter instead of always initializing a fresh one. `HFLogitProvider`/`load_local_engine`
  gain `adapter_path` to actually serve a trained LoRA adapter over its base model (previously,
  nothing in the serving path could load one). `POST /v1/models/load` (admin-only) loads a completed
  fine-tune artifact into the live registry without a gateway restart — the missing link between
  "trained a LoRA adapter" and "it's actually callable at `/v1/chat/completions`".
- A Next.js frontend (`frontend/`): a chat UI, and a runtime/knowledge-transport console landing
  page that live-checks `/health` and `/v1/models`, browses the live model registry, and diagrams
  the gateway's capability graph — labeling each node and edge as runtime-verified, source-known, or
  a design contract, never blurring the three.
- A full Sphinx manual (installation, quickstart, overview, package map, operator runbook, gateway
  operating contract and capabilities, validation, troubleshooting, security/data handling, release
  notes) plus generated API reference pages for every public module and route.

### Fixed

- The declared `mixle` dependency floor (`mixle>=0.2`) predated `mixle`'s own first PyPI release
  and was missing `mixle.evolve`, `mixle.task` (+ submodules), `mixle.ops.product_of_experts`, and
  `mixle.enumeration.AutoregressiveEnumerable` — all imported unconditionally. Verified against
  every published `mixle` release; the real floor is `mixle>=0.6.1`.
- The rented-GPU training path (`mixle-mlops train --backend mixle`, no `--local`) defaulted to
  installing mixle core from a git branch (`@evolve`) that no longer exists (merged, then deleted)
  — reproduced the resulting `pip install` failure. Now defaults to the pinned PyPI release.
  Same root issue fixed in the Docker images (`deploy/Dockerfile`, the root `Dockerfile`,
  `docker-compose.yml`), which defaulted to an unpinned `mixle@main` git install instead of the
  pinned PyPI dependency.
- `storage/objectstore.py`'s local (`file://`) backend required `fsspec`, which was only declared
  under the `cloud` extra — so a base install couldn't use the local object store at all, despite
  the module's own docs promising "no extra deps" for the local case. `fsspec` is now a base
  dependency; only the per-cloud drivers (`s3fs`/`gcsfs`/`adlfs`/`ossfs`) stay in `cloud`.
- `TreeLogitProvider`'s KV-cache trie crashed under `transformers>=5` (`IndexError` in GPT-2's
  position embedding): `transformers` 5 removed `Cache.to_legacy_cache`/`DynamicCache.
  from_legacy_cache`, and the fallback silently returned the *live, mutable* Cache object instead
  of an independent copy, so sibling tree branches corrupted each other's KV state. Rebuilt the
  fresh-copy path from `Cache.layers[i].keys/.values` (the stable public shape in transformers 5)
  for that version, while leaving the transformers<5 path untouched.
- `numpy` and `scipy` were imported directly and unconditionally (20+ modules) but never declared —
  only present by way of `mixle`'s own transitive dependency on them. Declared directly. `diffusers`
  (local Stable Diffusion) was imported but not declared under any extra at all; added an `image`
  extra.
- `mixle_mlops/__init__.py`'s `__version__` and the FastAPI app's reported `version` were both
  hardcoded at the stale `0.1.0`; both now track `pyproject.toml`.
- A cloud deployment (`MIXLE_DEPLOYMENT=cloud`) could boot silently on the default, well-known
  `dev-insecure-change-me` secret key (the pepper for password hashing and the OAuth device-flow
  HMAC) if an operator forgot to set `MIXLE_SECRET_KEY`. Now refuses to start, mirroring the
  existing `MIXLE_DATABASE_URL` guard.
- Removed `mixle_mlops/app.py` (a vestigial single-model server pre-dating the platform gateway,
  unreferenced anywhere else in the repo) and its paired root `Dockerfile`, whose env vars only
  applied to that dead module while its `CMD` actually launched the real gateway.
- The README's self-evolution note claimed `mixle.evolve` was only available on an unmerged branch;
  it has shipped on PyPI since `mixle` 0.6.1.
