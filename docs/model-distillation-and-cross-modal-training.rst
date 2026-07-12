Model Distillation and Cross-Modal Training
===========================================

``mixle-mlops`` is the right package for serving and operating distilled
models, cross-modal training jobs, feedback loops, and promotion gates. Core
Mixle owns probabilistic model contracts; MLOps owns the gateway, registry,
dataset export, training records, and deployment evidence.

Distillation Workflow
---------------------

A distillation run should move through explicit stages:

``source task``
    The teacher model, prompt, simulator, solver, or ensemble that creates
    labels, rationales, scores, or preference data.

``student target``
    The model or route being trained to approximate the source behavior.

``dataset export``
    Versioned examples with input, target, metadata, source receipt, and data
    license or synthetic-data status.

``training record``
    Parameters, environment, code version, data version, metrics, and produced
    artifacts.

``evaluation``
    Held-out metrics, calibration checks, regression tests, and domain-specific
    failure cases.

``promotion``
    A reviewed decision that links the training record, evaluation evidence,
    rollback plan, and deployment target.

No route should silently switch to a distilled model just because a training
job completed. Promotion is an explicit registry decision.

Multiple-Task Distillation
--------------------------

For multiple tasks, record task identity as first-class metadata:

* task name and version;
* input schema;
* output schema;
* loss or scoring function;
* teacher/source route;
* sampling policy;
* evaluation split;
* task weight in joint training.

A multitask result should report per-task metrics and aggregate metrics. A
single aggregate score is not enough because one task can improve while
another regresses.

Cross-Modal Training
--------------------

Cross-modal workflows should keep modality boundaries visible:

``text``
    Prompts, documents, retrieval snippets, chat turns, structured labels.

``image``
    Image references, transforms, generated outputs, captions, and safety
    metadata.

``tabular``
    Feature schemas, missing-value policy, normalization, and provenance.

``simulation or fields``
    PDE outputs, grids, sensor observations, posterior summaries, and
    simulator settings.

Training data should store modality-specific metadata and a joined example ID.
If one modality is synthetic or generated, label it at the example level.

Gateway and Registry Responsibilities
-------------------------------------

The gateway should expose stable routes for:

* dataset export;
* training job creation and status;
* evaluation result retrieval;
* registry candidate creation;
* promotion approval or rejection;
* deployment status and rollback.

The registry should keep aliases, candidate versions, approval decisions, and
rollback targets separate. Tests should prove that an unapproved candidate
cannot become the production alias.
