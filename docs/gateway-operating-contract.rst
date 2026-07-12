Gateway Operating Contract
==========================

The ``mixle-mlops`` gateway hosts Mixle models, LLM providers, datasets,
feedback loops, RAG, multimodal files, MCP tools, and training jobs behind one
server boundary. This page documents the expected behavior of that boundary so
clients, demos, notebooks, and mobile apps do not need to know internal
storage or registry details.

Route Families
--------------

.. list-table::
   :header-rows: 1

   * - Family
     - Responsibility
   * - ``/v1/chat/completions``
     - OpenAI-compatible chat path with rate limiting, multimodal
       normalization, optional RAG augmentation, response caching, provider
       dispatch, and conversation persistence.
   * - ``/v1/mixle/predict``, ``/score``, ``/latent``, ``/decide``
     - Distribution and decision routes over registered Mixle-capable models.
   * - ``/v1/mixle/capabilities/{model_id}``
     - Capability discovery for a registered model.
   * - ``/v1/datasets/*``
     - Dataset generation, code-task data, metadata persistence, and blob-store
       export.
   * - ``/v1/fine_tunes/*``
     - Structured local fine-tune jobs and offline GPU plan generation
       (including LoRA/QLoRA SFT and continuation runs).
   * - ``/v1/models/load``
     - Load a completed fine-tune artifact into the live model registry.
       Admin-only -- it changes what every caller sees at ``/v1/models``.
   * - ``/v1/rag/*`` and ``/v1/documents``
     - Document upload, parsing, indexing, and retrieval.
   * - ``/v1/pool/*``
     - Pool job submission, spend state, and artifact retrieval.
   * - ``/v1/substrate/{name}/*``
     - Knowledge substrate retrieval, budgeted context assembly, factuality
       receipts, and publish/propose/approve/reject promotion review.
   * - ``/v1/create``, ``/v1/uq``, ``/v1/simulate``, ``/v1/synthesize``,
       ``/v1/skills``, ``/v1/lineage/{model_id}``
     - Creation-verb twins of core Mixle operations (certified fit, honest
       uncertainty, constrained synthesis), skill registration, and
       data-to-model-to-skill lineage queries.
   * - ``/v1/telemetry/*``
     - PII-free platform events and training rows for router/placement
       improvement.
   * - ``/mcp``
     - Model-registry-backed MCP tool listing and tool calls, a single
       JSON-RPC 2.0 endpoint. Unlike the other route families, this path is
       not under ``/v1``.

Authentication and Ownership
----------------------------

Routes that read or mutate user state should depend on ``require_user``. User
scoped data must stay scoped by user id:

* conversations;
* files and document indexes;
* generated datasets;
* fine-tune jobs;
* pool jobs and spend records;
* substrate shards, published items, and pending promotions;
* created artifacts, registered skills, and lineage records;
* telemetry rows;
* feedback records.

Anonymous or local-only paths should be explicit in tests and docs. A route
must not silently switch from user-scoped persistence to process-global state
without calling that out in release notes.

Model Resolution
----------------

Model-serving routes resolve model ids through the application registry. The
expected failure states are:

``404``
    The requested model id is not registered.

``422``
    The model exists but lacks the requested Mixle capability. For example,
    ``/v1/mixle/latent`` should surface a capability error instead of returning
    a malformed response.

``429``
    The request exceeded the configured rate limit.

``5xx``
    Provider, storage, or uncaught infrastructure failure.

Clients should use ``/v1/models`` and
``/v1/mixle/capabilities/{model_id}`` to discover what a model can do before
calling specialized Mixle routes.

Decision Routes
---------------

``/v1/mixle/decide`` accepts a candidate action set and a named scalar loss.
The HTTP route intentionally accepts named losses, not arbitrary Python
callables. Built-in losses include:

``squared``
    Squared-error action selection.

``absolute``
    Median-optimal absolute-error action selection.

``linex``
    Asymmetric LINEX loss using ``loss_params["c"]``.

``newsvendor``
    Underage/overage cost with ``loss_params["cu"]`` and
    ``loss_params["co"]``.

Custom callables belong in in-process adapters or trusted worker code, not in
public HTTP payloads.

Training Job Lifecycle
----------------------

Fine-tune jobs move through explicit states:

``queued``
    The request was accepted and stored.

``running``
    A worker is training or labeling.

``succeeded``
    The trained artifact was written, metrics were recorded, and the live
    registry was updated for that model id.

``failed``
    The worker recorded an exception or invalid input as job data.

``planned``
    A GPU plan was produced without spending money or launching the provider.

``cancelled``
    The job was cancelled before completion.

Workers should update the row rather than raising into the route caller. A bad
training request is operational data, not a gateway crash.

Artifact and Storage Policy
---------------------------

Routes that generate bytes should return a stable artifact reference instead
of embedding large outputs directly. Local deployments may use filesystem or
SQLite-backed stores; cloud deployments may use object stores or managed
databases. The route response should make clear which state is durable and
which state is cache-only.

Generated artifacts should record:

* model or route that produced them;
* source data or prompt fingerprint;
* seed and parameters when applicable;
* blob or file location;
* media type;
* owner/user id;
* validation or verification status when available.

Promotion and Deployment
------------------------

Training, dataset generation, and telemetry can produce candidates. They should
not silently mutate production aliases. Promotion requires a distinct decision
record that links:

* candidate artifact;
* evaluation evidence;
* reviewer or policy result;
* rollback target;
* deployment alias or route.

This keeps ``observed``, ``candidate``, ``approved``, and ``deployed`` as
separate states in operational docs and API behavior. See :doc:`validation`
for how to test route behavior.

``POST /v1/models/load`` is this pattern's ``deployed`` step for an ``llm``
fine-tune: the candidate (a trained LoRA/QLoRA adapter, already recorded
under ``registry_root``) becomes servable only when an admin explicitly
loads it, not the moment training finishes.

