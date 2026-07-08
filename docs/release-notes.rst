Release Notes
=============

``mixle-mlops`` is the serving and operations package for the 0.6.3 family. It
documents gateway capabilities, model/provider surfaces, substrate serving,
telemetry, pool jobs, creation verbs, and drift retraining.

Included
--------

* Sphinx manual with operator runbook, package map, API overview, validation,
  and troubleshooting pages.
* gateway capability guide.
* Generated API pages for public modules and routes.
* Documentation extra in package metadata.
* ``docs/_build`` ignore rule for local builds.

Validation Evidence
-------------------

Record:

* focused tests for touched routes/services;
* local route smoke checks when server behavior changes;
* backend configuration used for storage/cache tests;
* whether external services were mocked, skipped, or genuinely exercised;
* ``python -m sphinx -W -b html docs docs/_build/html``.

Known Risks
-----------

* Route tests can pass against in-memory state while production uses Redis,
  databases, or object stores.
* Demo scripts should not mutate deployment aliases or registry state.
* Feedback/drift flows must preserve observed, candidate, approved, and
  deployed states separately.
