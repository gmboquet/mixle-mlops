Changelog
=========

This page tracks changes to the manual itself. For what shipped in each
package release, see the root `CHANGELOG.md
<https://github.com/gmboquet/mixle-mlops/blob/main/CHANGELOG.md>`_.

0.6.3
-----

* Added this Sphinx manual: installation, quickstart, overview, package map,
  operator runbook, gateway operating contract, gateway capabilities,
  model-distillation and cross-modal training, security and data handling,
  validation, troubleshooting, example execution manifest, release notes, and
  release readiness.
* Added the generated API reference for public ``mixle_mlops`` modules and
  gateway routes.
* Documentation now names the gateway, registry, auth, RAG, multimodal, MCP,
  telemetry, pool jobs, feedback, drift, and deployment boundaries explicitly
  instead of deferring to core Mixle's docs.
* ``docs/_build`` is ignored for local builds.
