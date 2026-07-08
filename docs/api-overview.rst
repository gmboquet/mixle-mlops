API Overview
============

The generated API reference is available in :doc:`api/modules`. This page maps
the larger package into operational areas.

Gateway
-------

``mixle_mlops.gateway.app``
    Application factory and route assembly.

``mixle_mlops.gateway.routes.*``
    HTTP routes for accounts, chat, files, models, datasets, RAG, MCP,
    feedback, tasks, solutions, pool jobs, substrate serving, telemetry, and
    creation verbs.

``mixle_mlops.gateway.auth`` and ``mixle_mlops.gateway.verifiers``
    Request authentication and verifier helpers.

Models And Providers
--------------------

``mixle_mlops.models.*``
    Local, echo, OpenAI-compatible, provider, Mixle-model, and task-cascade
    surfaces.

``mixle_mlops.core.*``
    Adapter, registry, predictive, and decision helpers that bridge Mixle
    capabilities into serving workflows.

Data, RAG, And Storage
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
