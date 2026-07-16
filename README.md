# mixle-mlops

The 0.8 operational kernel gives all Mixle semantic owners one provider-neutral execution boundary: owner-scoped
artifacts, immutable invocation identity, jobs and runs, leases, idempotency, bounded events, cancellation,
checkpoints, retry/recovery, result receipts, model candidates, evidence-gated promotion, deployments, and rollback.
Operational success is stamped `not_evaluated`; it never implies that a scientific result is correct.

`DurableLocalRunner` is the single-node reference implementation. It survives process restart, detects expired leases
and checkpoint tampering, preserves semantic hashes across retries, and isolates organization/project data.
`DeploymentRegistry` accepts immutable candidates only when exact trusted factory and harness receipts cover the same
artifact and required suites; failed deployments roll back through an auditable registry event.
`check_registry_integrity` (`mixle_mlops.control.integrity`) is a read-only auditor over that registry: it replays the
receipt log and compares it against the live alias state, flags any dangling alias or "previous" pointer, and --
when handed an artifact store -- verifies every candidate's artifact is still present and byte-exact. It never mutates
the registry; it only tells you whether the bookkeeping a crash or a hand-edited file could have disturbed still
agrees with itself.

The packaged `control_surface_manifest.json` classifies the older broad runtime as supported, partial, experimental,
compatibility, or application-specific and assigns migration owners. The gateway and integrations below remain useful,
but their presence is not a blanket maturity claim.

An **all-in-one AI platform**: host [mixle](https://github.com/gmboquet/mixle) probabilistic models *and* open
LLMs (Llama, DeepSeek, Qwen, …) behind one **OpenAI-compatible** gateway — with accounts, API keys, a chat UI,
multimodal, RAG, an MCP server, tool calling + a server-side agentic loop, and the things that make mixle more
than an LLM proxy: a **probabilistic-bridge stack** that lifts a laptop-sized model toward frontier quality, and
**self-evolution** that improves served models from their own usage.

It runs end-to-end on a laptop (SQLite + filesystem + a local Ollama) and scales to Postgres + object store +
Redis via config; the cloud blob backend (`S3BlobStore`) is a roadmap item (D4) — until it lands, cloud
multimodal/RAG upload is not wired. <!-- re-expand when D4/D2 land --> Local-first today; the database and
cache legs of "cloud" are config-only, the blob-storage leg isn't yet.

```sh
cp deploy/.env.example deploy/.env          # set MIXLE_SECRET_KEY
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d gateway ollama
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec ollama ollama pull llama3.2
curl localhost:8000/v1/models               # OpenAI-compatible
```

## What's in the box

| Capability | Surface |
|---|---|
| **OpenAI-compatible chat** (streaming) over any hosted model | `POST /v1/chat/completions`, `GET /v1/models` |
| **Host open LLMs** through their standard server (Ollama/vLLM/llama.cpp/hosted) | `MIXLE_LLM_BACKENDS` (local + cloud in one registry) |
| **Host mixle models** with a real probabilistic surface | `POST /v1/mixle/{predict,score,latent,decide}` |
| **Tool calling + server-side agent loop** (executes MCP tools, RAG, mixle decide/predict, exact compute) | `extra.agent`, `/mcp` |
| **Accounts, API keys, OAuth** (Sign in with Google/Apple) | `/auth/*`, `mk-…` keys |
| **Multimodal** (image inputs, local blob store only — cloud blob backend is roadmap D4), **RAG** (text-layer PDF/DOCX/PPTX + plaintext/markdown retrieval today, needs the `documents` extra — not installed by base; OCR/scanned-PDF/table/LAS parsing is roadmap D2), **image gen**, **dataset gen** <!-- re-expand when D4/D2 land --> | `/v1/files`, `/v1/documents`, `/v1/rag/search`, `/v1/images/generations`, `/v1/datasets` |
| **Conversations** (persisted threads + json/markdown/pdf export) | `/v1/conversations` |
| **Caching + rate limiting** (memory / Redis), **MCP server** | `extra` flags, `MIXLE_REDIS_URL`, `/mcp` |
| **Chat UI** (Next.js, Claude/ChatGPT-like) | `frontend/` |
| **Multi-cloud deploy** (AWS/Azure/GCP/Alicloud) | Helm chart + Terraform + `mixle-mlops init-cloud` |
| **Universal cloud compute** (GPU VMs, marketplaces, managed k8s, on-prem) | Any OpenAI-compatible endpoint + generic Docker/Kubernetes GPU recipes |

### Delivered vs roadmap

Every headline above maps to shipped code or an explicit task ID — nothing here is aspirational:

| Claim | Status |
|---|---|
| Chat/model serving, accounts + API keys + OAuth, tool calling + agent loop, caching/rate limiting, MCP server, chat UI, conversations export | shipped |
| Cloud config for database (Postgres) + cache (Redis); multi-cloud Helm/Terraform; universal cloud compute (any OpenAI-compatible endpoint) | shipped |
| Cloud blob backend (`S3BlobStore`, needed for multimodal/RAG upload under a cloud deployment) | roadmap — D4 |
| RAG: text-layer PDF/DOCX/PPTX + plaintext/markdown retrieval (requires the `documents` extra) | shipped |
| RAG: OCR/scanned-PDF, table extraction, LAS, page-level provenance | roadmap — D2 |

<!-- re-expand when D4/D2 land -->

## The mixle bridge — frontier quality on a laptop

mixle is the calibrated **judge / combiner / router** around a small local generator. None of this puts 1T
parameters on your laptop; it trades inference-time compute + verification + routing for quality on
**verifiable / computational / structured / retrieval-grounded** tasks (and escalates the rest). All opt-in per
request via `extra`:

| Lever | Request | What it does |
|---|---|---|
| **Best-of-N self-consistency** | `extra.best_of_n: 5` | sample N, return the majority answer + a calibrated confidence (`X-Self-Consistency`) |
| **Cascade router** (FrugalGPT) | `extra.cascade: {frontier, threshold, n}` | answer locally when confident, escalate to a frontier model only on the hard tail (`X-Cascade-Escalated`) |
| **Mixture-of-Agents** | `extra.moa: {proposers, aggregator}` | several models propose, an aggregator synthesizes |
| **Program-offload** (PAL) | the `mixle_solve` tool (agent mode) | the model offloads exact arithmetic/probability/stats to a deterministic solver instead of doing mental math |

## Self-evolution — the system improves itself

Served mixle models improve from data and from the platform's own usage, **only ever promoting a challenger that
verifiably and non-regressively beats the champion** (built on `mixle.evolve`):

- `POST /v1/evolve/{model}` — run measure → propose → verify → promote (auto-select / online-update / recalibrate / refit), with rollback.
- `POST /v1/evolve/tick` — one autonomous pass over every hosted mixle model.
- `GET /v1/evolve/{model}/signals` — the cascade router **self-calibrates** its threshold from observed traffic.
- Every cascade decision + best-of-N vote is recorded as in-distribution feedback the loop consumes.

## Develop

```sh
pip install -e ".[dev]"                      # + extras: documents, scale, datasets, export, cloud, mcp, all
                                              # documents = text-layer PDF/DOCX/PPTX RAG upload (not in base);
                                              # OCR/scanned-PDF/table/LAS parsing isn't in any extra yet (roadmap D2)
pytest -q                                    # the full suite
ruff check mixle_mlops tests
mixle-mlops create-user you@example.com pw --admin   # a user + an API key
mixle-serve                                  # the gateway on :8000 (or: uvicorn mixle_mlops.gateway.app:app)
cd frontend && npm install && npm run dev    # the chat UI on :3000
```

Config is env-driven (prefix `MIXLE_`, see `deploy/.env.example` and `mixle_mlops/config.py`). Key knobs:
`MIXLE_LLM_BASE_URL` (Ollama by default), `MIXLE_LLM_BACKENDS` (per-model local+cloud), `MIXLE_DATABASE_URL`
(sqlite→postgres), `MIXLE_OBJECT_STORE_URL` (file→s3/gcs/azure/oss), `MIXLE_REDIS_URL`, `MIXLE_REQUIRE_AUTH`.

## Deploy

```sh
mixle-mlops init-cloud aws                    # scaffold a provider-correct .env
# Terraform provisions bucket + Postgres + Redis; Helm runs the same image on any k8s:
cd deploy/terraform/aws && terraform apply
helm install mixle deploy/helm/mixle-mlops -f deploy/helm/mixle-mlops/values-aws.yaml
```

The same image + chart run on EKS/AKS/GKE/ACK — only three URLs (database / object store / redis) change. Compute is
provider-neutral: point `MIXLE_LLM_BASE_URL` / `MIXLE_LLM_BACKENDS` at any OpenAI-compatible server running on a GPU
VM, marketplace instance, managed Kubernetes cluster, or on-prem host. See `deploy/cloud/README.md` and
`deploy/compute/README.md`.

## Design

The split mirrors mixle's: **mixle** owns the domain-neutral math (distributions, scoring, calibration, decision,
`mixle.evolve`); **mixle-mlops** owns serving/orchestration (gateway, accounts, RAG, the bridge components, the
evolution worker). The mixle math always upstreams to the core. See `ARCHITECTURE.md`.

> Requires `mixle.evolve` (currently on the mixle `evolve` branch) for the self-evolution routes.
