Gateway Capabilities
====================

``mixle-mlops`` is the gateway and operations package for Mixle. Core modeling
stays in ``mixle`` and sibling packages; this package owns authentication,
persistence, queues, registry boundaries, route registration, spend gates, and
deployment-adjacent service behavior.

Scope
-----

This guide documents these gateway surfaces:

* substrate serving for retrieval, context assembly, factuality receipts,
  publishing, and promotion review;
* telemetry ingestion for PII-free decision events and training rows;
* pool job submission with budget, confirmation, quota, spend, and artifact
  round-trip checks;
* creation verbs for certified model artifacts, uncertainty quantification,
  simulation, verified synthesis, skills, and lineage;
* Mixle model-serving manifests with certificate and calibration metadata;
* drift retraining that promotes only after a drift-triggered refit;
* LoRA/QLoRA fine-tuning (SFT on chat data or raw continuation, plus resuming
  from a prior adapter) and loading a completed fine-tune into the live model
  registry.

Route Families
--------------

.. list-table::
   :header-rows: 1

   * - Area
     - Route family
     - Contract
   * - Substrate
     - ``/v1/substrate/{name}``
     - Inspect persisted shards, ingest typed evidence, retrieve cited items,
       assemble context, check factuality, and manage publish/propose/review
       flows.
   * - Telemetry
     - ``/v1/telemetry``
     - Persist typed decision events and extract training rows for learned
       routing or placement policies.
   * - Pool
     - ``/v1/pool/jobs`` and ``/v1/pool/spend``
     - Submit budgeted jobs, enforce confirmation and quota rails, and return
       completed artifacts with provenance.
   * - Creation verbs
     - ``/v1/create``, ``/v1/uq``, ``/v1/simulate``, ``/v1/synthesize``
     - Store certified model artifacts, quantify uncertainty when available,
       simulate records, and generate verifier-checked synthetic data.
   * - Skills and lineage
     - ``/v1/skills`` and ``/v1/lineage/{model_id}``
     - Register model-backed skills and report data-to-model-to-skill lineage.
   * - Fine-tuning and model loading
     - ``/v1/fine_tunes`` and ``/v1/models/load``
     - Plan/train a LoRA or QLoRA fine-tune (SFT or continued-pretraining
       style), then load the resulting adapter into the live registry.

Substrate Serving
-----------------

Substrate endpoints expose evidence retrieval and review state. The
``/context`` route returns context, citations, and an abstain decision; it does
not generate an answer. Callers must either answer from the returned evidence
or respect ``abstain: true``.

Publishing and promotion are explicit. A user may publish or propose items
when authorized, but curated promotion requires an approval endpoint. Retrieval
must not make visibility or promotion a side effect.

Telemetry
---------

Telemetry events are decision records, not raw content. They should carry a
kind, feature dictionary, choice, outcome, and tags. Unknown event kinds are
rejected per event rather than crashing the batch.

Training-row extraction is evidence for learned orchestration policies. Static
policies remain the default until a learned policy is receipted as no worse on
the recorded history.

Pool Jobs and Spend Rails
-------------------------

Pool endpoints enforce budget and quota before work starts:

* estimated cost above budget rejects the job;
* priced jobs require explicit confirmation;
* quota overflow rejects before execution;
* rejected jobs cost nothing;
* completed fit jobs return a serialized artifact and canonical parameter
  fingerprint.

Unsupported operation kinds should be recorded as rejected outcomes, not
opaque worker failures.

Creation, Model Manifests, and Drift
------------------------------------

Creation routes are HTTP twins of core Mixle artifact operations. Stored
models should record certificate, calibration status when requested, strategy,
row count, data fingerprint, and parameter fingerprint.

``MixleAdapter.info()`` should expose certificate and calibration metadata when
the hosted model provides it. Clients should be able to gate on estimation and
uncertainty evidence, not only model identifiers.

Drift retraining compares recent traffic against the production reference. If
drift is detected, it refits, registers, promotes a new production version, and
rolls the reference forward. If no drift is detected, the production alias is
untouched.

Focused tests for these surfaces are named in :doc:`validation`; run them
together before changing any route family described above.

Fine-Tuning: LoRA, QLoRA, SFT, and Continuation
------------------------------------------------

``POST /v1/fine_tunes`` with ``backend: "llm"`` returns an offline training
*plan* for a LoRA (or, with ``qlora: true``, 4-bit QLoRA) fine-tune of a
HuggingFace causal LM -- it does not spend money or launch a GPU itself.
Actually running it (rented vast.ai box or ``--local``) is the operator's
keyed ``mixle_mlops.compute.launch``/``run_local`` step, same as before.

The generated training script picks its objective from the dataset shape:

* a ``messages`` column (chat turns) applies the base model's chat template
  and masks the prompt out of the loss, so only the assistant's response
  contributes -- supervised fine-tuning (SFT), not raw continuation.
* a ``text`` column (or no ``messages`` column at all) trains on the raw text
  unmasked, the same behavior this script has always had -- usable as a
  continued-pretraining-style (CPT) run over unlabeled text.
* ``resume_from`` (a prior adapter directory) continues training that
  adapter instead of initializing a fresh one, for incremental/CPT-style runs
  that pick up where a previous pass left off.

Once an artifact is trained and registered under ``registry_root`` (writing
the ``metadata.json`` the launcher already produces), ``POST
/v1/models/load {"name": ...}`` loads it into the *live* gateway registry --
no restart required. This is admin-gated, like substrate's promotion
approval: it doesn't train or spend anything, but it does change what every
caller sees at ``/v1/models``, so it isn't a self-service action. Today it
only serves the ``llm`` backend (a base model plus an optional LoRA
adapter, loaded through the same local logit-level engine that hosts any
other local model); a completed ``structured``/``mixle``-backend artifact is
registered through its own existing path instead.

There is no per-request, per-user adapter switching (serving many users'
distinct adapters concurrently off one base model, the way vLLM's or
S-LoRA's multi-LoRA serving does) -- a "per-user model" today means loading
that user's adapter as its own named model via ``/v1/models/load`` and
routing to it by id, which the registry already supports for free. Dynamic
multi-tenant adapter serving would be a different serving engine, not an
extension of this one, and isn't built here.

Boundaries
----------

The gateway persists artifacts and queues under ``registry_root``. Production
deployments should place that root on durable storage. New operation kinds must
be explicit tested executors with budget and spend rails.
