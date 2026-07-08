Release Readiness
=================

``mixle-mlops`` is the operational package for serving, gateway routes,
registry state, accounts, feedback, storage, datasets, and deployment helpers.
Release readiness means local tests pass and the operational state transitions
are documented clearly enough to avoid accidental production mutation.

Supported Environment
---------------------

The package metadata declares Python 3.10 and newer. Release evidence should
record the backend configuration used for each check: in-memory, SQLite,
Redis, object store, local model runtime, cloud service, or mocked provider.

Service Gates
-------------

For route or service changes, record:

* focused tests for the touched route, service, or data model;
* local gateway smoke command and response shape;
* authentication mode and credential source;
* storage/cache backend used during the check;
* expected artifacts, logs, or receipts; and
* whether external providers were mocked, local, or genuinely exercised.

Promotion And Registry Gates
----------------------------

Deployment-affecting workflows must preserve observed, candidate, approved,
and deployed states separately. A demo, feedback run, or local training script
should not write production aliases directly. Promotion decisions should be
auditable and reversible through documented state transitions.

Documentation Gates
-------------------

The operator runbook, gateway contract, example execution manifest, and API
reference should match the shipped route surface. Build Sphinx with warnings
as errors and from a clean archive before release.
