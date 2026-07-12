# Deploying the mixle-mlops platform gateway

These assets deploy the **platform gateway** — the FastAPI app
(`mixle_mlops.gateway.app:app`, run by `mixle-serve`) that exposes the OpenAI-compatible API
(`/v1/chat/completions`, `/v1/models`, …) plus the mixle distribution/decision and feedback
routes — fronting the model registry: mixle probabilistic models **and** open LLMs (via any
OpenAI-compatible backend, Ollama by default).

> This supersedes the old `k8s/` manifests, which served the single mixle-model scoring server.
> The fresh, gateway-oriented manifests live in [`k8s/`](./k8s/).

## Layout

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage, slim, non-root image. Builds the package wheel, installs it + the mixle core (from git), runs `mixle-serve` on `0.0.0.0:8000`. |
| `docker-compose.yml` | Local stack: `gateway` + `ollama` (default LLM backend), with an optional `postgres` (`--profile cloud`) and a named `mixle_data` volume. |
| `docker-compose.gpu.yml` | Optional NVIDIA GPU override for single-host GPU compute. |
| `.env.example` | All documented `MIXLE_*` env vars. Copy to `.env`. |
| `k8s/` | `Deployment`+`Service` for the gateway (env from ConfigMap/Secret), a `PVC` for `mixle_data`, and an Ollama `Deployment`+`Service`. |
| `compute/` | Provider-neutral recipes for GPU VMs, marketplace instances, managed k8s, and external OpenAI-compatible endpoints. |

## Run locally with Docker Compose

```bash
# 1. Configure
cp deploy/.env.example deploy/.env
# edit deploy/.env — at minimum set a real MIXLE_SECRET_KEY:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"

# 2. Start the gateway + Ollama (builds the image on first run)
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d gateway ollama

# 3. Pull an LLM into Ollama (must match MIXLE_LLM_MODELS)
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec ollama ollama pull llama3.2

# 4. Verify the gateway is up
curl http://localhost:8000/health
```

### Call the OpenAI-compatible API

List models (the always-available `echo` model needs no backend):

```bash
curl http://localhost:8000/v1/models
```

Chat completion against the echo model (no auth needed if `MIXLE_REQUIRE_AUTH=false`; otherwise
create a key first — see below):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MIXLE_API_KEY" \
  -d '{"model":"echo","messages":[{"role":"user","content":"hello"}]}'
```

Chat against the LLM backend (after the Ollama pull above):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MIXLE_API_KEY" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

### Get an API key

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec gateway \
  mixle-mlops create-user you@example.com hunter2
# prints an api key — export it as MIXLE_API_KEY for the curls above
```

### Cloud mode (Postgres)

```bash
# in deploy/.env:
#   MIXLE_DEPLOYMENT=cloud
#   MIXLE_DATABASE_URL=postgresql+psycopg://mixle:mixle@postgres:5432/mixle
docker compose -f deploy/docker-compose.yml --profile cloud --env-file deploy/.env up -d
```

### GPU host mode

On any NVIDIA GPU VM or rented marketplace box with Docker and the NVIDIA container runtime:

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml \
  --env-file deploy/.env up -d gateway ollama
```

This is still provider-neutral: the gateway only requires that the model backend expose an OpenAI-compatible `/v1`
API. See [`compute/README.md`](./compute/README.md) for external vLLM/Ollama/llama.cpp/TGI endpoints and Kubernetes
GPU scheduling.

## Run on Kubernetes

```bash
# 1. Create the secret (don't apply secret.yaml's placeholder in prod)
kubectl create secret generic mixle-gateway-secret \
  --from-literal=MIXLE_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"

# 2. Apply config + storage + workloads
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/ollama.yaml
kubectl apply -f deploy/k8s/gateway.yaml

# 3. Pull a model into Ollama
kubectl exec deploy/mixle-ollama -- ollama pull llama3.2

# 4. Reach the gateway
kubectl port-forward svc/mixle-gateway 8000:80
curl http://localhost:8000/health
```

Build & push the image first, and set `image:` in `k8s/gateway.yaml`:

```bash
docker build -f deploy/Dockerfile -t <your-registry>/mixle-mlops:latest .
docker push <your-registry>/mixle-mlops:latest
```

### Scaling notes

Local mode keeps state (SQLite + registry) on a `ReadWriteOnce` PVC, so the gateway runs as a
single replica with a `Recreate` strategy. To scale horizontally, switch to **cloud mode**
(`MIXLE_DEPLOYMENT=cloud` + a Postgres `MIXLE_DATABASE_URL` + S3) — the gateway then becomes
stateless and you can raise `replicas` / attach an HPA and drop the PVC mount.

## Notes

- The image installs the mixle core from PyPI by default (the pinned `mixle>=0.6.1` dependency in
  `pyproject.toml`). Pass `--build-arg MIXLE_GIT="git+https://github.com/gmboquet/mixle@<ref>"` only to
  test against an unreleased mixle branch or commit.
- `MIXLE_LLM_MODELS` is a JSON list (e.g. `["llama3.2","qwen2.5"]`); the ids must be pulled in
  Ollama (or available on whatever OpenAI-compatible backend `MIXLE_LLM_BASE_URL` points at).
- Ollama is just the default OpenAI-compatible backend. Point `MIXLE_LLM_BASE_URL` at a vLLM,
  llama.cpp, TGI, or hosted endpoint to swap it out without touching the gateway.
- Compute providers do not need first-class adapters if they can run one of those backends. Use
  `MIXLE_LLM_BACKENDS` when several GPU pools or providers should appear in one model registry.
