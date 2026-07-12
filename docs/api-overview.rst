API Overview
============

The generated API reference is available in :doc:`api/modules`. It covers the
public modules and their docstring-documented interfaces; internal helpers
without docstrings are intentionally kept out of the published reference. This
page maps the larger package into operational areas.

Gateway
-------

``mixle_mlops.gateway.app``
    Application factory and route assembly.

``mixle_mlops.gateway.routes.*``
    HTTP routes for accounts, chat, files, models, datasets, RAG, MCP,
    feedback, tasks, solutions, pool jobs, substrate serving, telemetry, and
    creation verbs. See :doc:`gateway-operating-contract` for the route
    lifecycle and failure-state expectations.

``mixle_mlops.gateway.auth`` and ``mixle_mlops.gateway.verifiers``
    Request authentication and verifier helpers.

Models and Providers
--------------------

``mixle_mlops.models.*``
    Local, echo, OpenAI-compatible, provider, Mixle-model, and task-cascade
    surfaces.

``mixle_mlops.core.*``
    Adapter, registry, predictive, and decision helpers that bridge Mixle
    capabilities into serving workflows.

Data, RAG, and Storage
----------------------

``mixle_mlops.datasets.*``
    Dataset model, export, generation, and code-task helpers.

``mixle_mlops.documents.parse``
    Optional document parsing for RAG ingestion.

``mixle_mlops.rag.*``
    Embeddings, indexes, vector stores, and retrieval augmentation.

``mixle_mlops.storage.*`` and ``mixle_mlops.cache.*``
    Database, object storage, response cache, Redis, memory cache, and rate
    limiting.

Learning Loops
--------------

``mixle_mlops.feedback.*``
    Feedback collection, elicitation, rewards, feature rewards, and training
    loop support.

``mixle_mlops.evolve.*``
    Lineage, model records, policies, schedulers, signals, and workers.

``mixle_mlops.drift_retrain``
    Drift-triggered retraining entry point.

Accounts, Conversations, and Access Control
-------------------------------------------

``mixle_mlops.accounts.*``
    Account models, OAuth helpers, device-code flows, service helpers, and
    security utilities. These modules define who can call the gateway and how
    API keys or account identities are represented.

``mixle_mlops.conversations.*``
    Conversation records, export helpers, and service functions for preserving
    chat state and review evidence across gateway calls.

``mixle_mlops.config`` and ``mixle_mlops.gateway.app``
    Configuration loading and application assembly. Review these modules when
    changing environment variables, storage paths, provider defaults, or server
    startup behavior.

Multimodal and Creation Surfaces
--------------------------------

``mixle_mlops.multimodal.*``
    Content records and storage helpers for image, text, and mixed payloads.
    Use these modules when a route needs to preserve modality metadata instead
    of flattening everything into text.

``mixle_mlops.image_gen.*``
    Image-generation adapter boundaries, including local diffusion integration
    points. Treat provider-backed image generation as an optional capability
    with explicit dependency and credential requirements.

``mixle_mlops.mcp.*``
    MCP client, server, and schema-bridge helpers for tool integration. These
    modules should preserve tool provenance, approval requirements, and
    request/response schemas.

Training and Promotion Review
-----------------------------

``mixle_mlops.training.*``
    Training service and model records used by distillation and fine-tuning
    workflows.

``mixle_mlops.seed_registry`` and ``mixle_mlops.core.registry``
    Registry and seed-model helpers. Review these modules with the promotion
    path: a trained or distilled artifact should not become the served default
    without recorded evaluation evidence and an explicit deployment decision.
