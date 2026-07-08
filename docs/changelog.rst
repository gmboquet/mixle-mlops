Changelog
=========

This changelog records documentation-visible release changes for
``mixle-mlops``.

0.6.3 Release Branch
--------------------

See :doc:`release-0-6-3` for scope, validation evidence, and known risks.

Added
~~~~~

* Sphinx manual for gateway operation, route behavior, package map, security,
  validation, troubleshooting, and example execution manifests.
* 0.6.3 gateway-capability guide covering substrate serving, telemetry, pool
  jobs, creation verbs, lineage, and drift/retraining.
* Model-distillation and cross-modal training guide for datasets, training
  jobs, registry decisions, and promotion evidence.
* Generated API reference for public modules and routes.
* Release-readiness checklist for route smoke checks, backend configuration,
  credentials, promotion state, and clean-archive docs evidence.

Changed
~~~~~~~

* Docs separate serving, registry, feedback, deployment, and demo boundaries
  so scripts do not bypass promotion decisions.
* The docs tree is Sphinx/reStructuredText only.

Release Gate
~~~~~~~~~~~~

A public release is not complete until focused route/service tests, local
server smoke checks, strict Sphinx docs, clean packaging checks, and the
coordinated family manifest all refer to the same commit.
