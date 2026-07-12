Package Map
===========

``mixle_mlops.gateway``
    FastAPI app factory, route modules, auth, verifiers, agent loop, routing
    policies, MoA, PoE, cascades, best-of-N, constrained decoding, and
    program offload.

``mixle_mlops.models``
    Echo, local, OpenAI-compatible, provider, Mixle-model, and task-cascade
    adapters.

``mixle_mlops.core``
    Capability adapters, registry, predictive helpers, and decision helpers.

``mixle_mlops.accounts`` and ``mixle_mlops.conversations``
    Users, API keys, OAuth/device-code flows, conversation storage, and export.

``mixle_mlops.rag``, ``mixle_mlops.documents``, and ``mixle_mlops.multimodal``
    Document parsing, embeddings, vector stores, retrieval augmentation,
    multimodal content, and file storage.

``mixle_mlops.feedback``, ``mixle_mlops.training``, and ``mixle_mlops.evolve``
    Feedback collection, reward modeling, fine-tune records, drift retraining,
    lineage, scheduler, signals, and workers.

``mixle_mlops.datasets`` and ``mixle_mlops.compute``
    Dataset generation/export and rented-GPU training job specs (LoRA/QLoRA
    fine-tune script generation, offline planning, launch).

``mixle_mlops.storage`` and ``mixle_mlops.cache``
    Database, object-store, response-cache, Redis, in-memory cache, and rate
    limiting.

``deploy`` and ``frontend``
    Deployment assets and the Next.js chat UI.

``docs/api``
    Generated API pages for gateway, provider, storage, account, feedback,
    training, RAG, MCP, multimodal, and route modules. Missing API pages hide
    operational behavior and should block public release review.

Ownership Boundaries
--------------------

Route modules own HTTP behavior and error mapping. Core modules own registry,
decision, and capability bridges. Model/provider modules own provider
adaptation and should not mutate deployment state directly. Storage and cache
modules own persistence semantics. Frontend code should expose those states
without becoming the source of truth for access policy, registry contents, or
deployment decisions. Keeping those boundaries visible helps a reviewer trace
a request from authentication through model execution, artifact capture,
feedback, and promotion evidence. Training helpers should write enough
metadata for ``mixle-aifactory`` and release review to understand where an
artifact came from and why it is safe to promote.

Cross-Package Boundary
-----------------------

When a workflow crosses package boundaries, keep the operational mutation in
``mixle-mlops`` and the scientific or demo logic in the owning sibling package.
For example, ``mixle-demos`` may produce a report, but registry mutation,
serving aliases, credentials, and rollback records belong here.
