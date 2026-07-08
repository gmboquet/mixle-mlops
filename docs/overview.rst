Overview
========

``mixle-mlops`` owns the operational layer around Mixle models. The package is
responsible for serving, persistence, account flows, provider bridges,
feedback, datasets, object storage, RAG, multimodal payloads, MCP integration,
and deployment scaffolding.

Package Boundaries
------------------

The package owns:

* FastAPI gateway construction and route registration.
* Provider adapters and OpenAI-compatible model interfaces.
* Accounts, API keys, conversations, files, datasets, and object storage.
* RAG indexes, document parsing, embeddings, and retrieval augmentation.
* Feedback collection, reward modeling, self-evolution signals, and drift
  retraining orchestration.
* Pool jobs, substrate routes, telemetry, creation verbs, lineage, and spend
  rails added for the 0.6.3 release branch.
* Deployment helpers for local, container, and cloud environments.

The package does not own Mixle's core probability models, PDE kernels, discrete
numerical methods, or notebook content. It adapts and serves those packages.

Primary Surfaces
----------------

``mixle_mlops.gateway``
    The HTTP application, route modules, tool registry, verifiers, and gateway
    orchestration helpers.

``mixle_mlops.models``
    Provider adapters, local engine surfaces, task cascades, and Mixle model
    registration.

``mixle_mlops.rag`` and ``mixle_mlops.documents``
    Retrieval, vector storage, embeddings, and document parsing.

``mixle_mlops.feedback`` and ``mixle_mlops.evolve``
    Feedback collection, reward extraction, lineage, policy, scheduling, and
    evolution workers.

``mixle_mlops.storage`` and ``mixle_mlops.cache``
    Database, object store, in-memory cache, Redis cache, response cache, and
    rate-limit primitives.
