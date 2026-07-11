# Changelog

All notable changes to `mixle-mlops` are documented here. This is the first versioned,
checklist-verified release; earlier history (84 commits building the platform from a single-model
scoring server into the full gateway) predates formal changelog tracking.

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
- A local logit-level engine (token-level PoE + grammar-masked constrained decoding via
  `transformers`), including a KV-cache trie (`TreeLogitProvider`) for efficient tree-structured
  decoding/enumeration.
- Multi-cloud deploy: Docker/Helm/Terraform for AWS/Azure/GCP/Alicloud, a unified `fsspec`-backed
  object store, and generic GPU-compute launch (rented boxes via vast.ai, or any OpenAI-compatible
  endpoint).
- A Next.js chat UI (`frontend/`).

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
