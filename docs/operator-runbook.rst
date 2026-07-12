Operator Runbook
================

``mixle-mlops`` is the serving and operations package for the Mixle family. It
owns gateway routes, account surfaces, storage, cache behavior, job launchers,
registry-like model metadata, feedback loops, and deployment helpers. This
runbook is for local release review and for operators trying to understand
which part of the package should own a workflow.

Local Bring-Up
--------------

Start by installing the package with the extras needed for the surface under
review:

.. code-block:: console

   python -m pip install -e ".[docs]"
   python -m pip install -e ".[datasets,structured,export]"

Use the project README for the current application command. Record any
environment variables used for the run, but never commit secrets or real API
keys in docs, examples, fixtures, or screenshots.

Gateway Surfaces
----------------

The gateway should be the boundary for:

* OpenAI-compatible chat/model endpoints;
* Mixle model manifests and local model adapters;
* substrate-style retrieve, reason, answer, and act routes;
* pool or job execution requests;
* telemetry and usage records;
* feedback and dataset export flows.

Do not make mobile, notebook, or demo packages mutate registry or deployment
state directly. They should call a documented server-side route or stay in
offline/demo mode.

The route-level operating contract is documented in
:doc:`gateway-operating-contract`.

Artifacts and Receipts
----------------------

Operational artifacts should be traceable. When a route produces or promotes
an artifact, record:

* request id or job id;
* source model or dataset;
* output location;
* verification status;
* reviewer or policy state;
* limitations and rollback path.

This should line up with ``mixle-knowledge`` receipt and verification
contracts rather than inventing a second audit language.

Drift and Feedback
------------------

Feedback, drift, and retraining routes are operationally sensitive. A release
review should distinguish:

``observed``
    Metrics or feedback were collected.

``candidate``
    A retraining or replacement candidate was created.

``approved``
    A reviewer or policy gate approved promotion.

``deployed``
    Serving state actually changed.

Do not collapse these states into a single "updated" flag.

Storage and Caches
------------------

Cache and storage backends should be replaceable. Local tests may use in-memory
or file-backed stores, while production deployments can use Redis, object
stores, or managed databases through optional extras.

When documenting a route, say which state is durable and which is cache-only.

Deployment Boundaries
---------------------

Deployment helpers can prepare infrastructure, containers, or job specs. They
should not hide irreversible changes behind demo scripts. Promotion into a
serving alias should flow through explicit decision records and audited
endpoints.
