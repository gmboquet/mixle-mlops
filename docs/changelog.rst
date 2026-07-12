Changelog
=========

This page tracks changes to the manual itself. For what shipped in each
package release, see the root `CHANGELOG.md
<https://github.com/gmboquet/mixle-mlops/blob/release/0.7.0/CHANGELOG.md>`_
(this link should move to ``main`` once the release branch merges).

0.7.0
-----

* Added a 0.7.0 entry to :doc:`release-notes` and resolved its stale 0.6.3
  "Known Risks" against what the 0.7.0 release-readiness pass actually
  verified.
* Filled two gaps in the "Route Families" table and the user-scoped-data list
  in :doc:`gateway-operating-contract`: the substrate and creation-verb/
  skills/lineage route families were documented in
  :doc:`gateway-capabilities` but missing from the operating contract itself.
* Fixed a copy-paste error in :doc:`quickstart` and :doc:`operator-runbook`
  that told a reader to install the ``docs`` extra (Sphinx tooling) to run
  the gateway, instead of ``dev``.
* Added the new ``image`` extra to the extras list in :doc:`installation`.
* Consolidated ``docs/release-readiness.rst`` into :doc:`validation` (the
  narrower, test-command-focused page); the standalone page no longer exists.
* Dropped the stale ``mixle_mlops.app`` API page and narrative references
  left over from that module's removal.

0.6.3
-----

* Added this Sphinx manual: installation, quickstart, overview, package map,
  operator runbook, gateway operating contract, gateway capabilities,
  model-distillation and cross-modal training, security and data handling,
  validation, troubleshooting, example execution manifest, and release notes.
* Added the generated API reference for public ``mixle_mlops`` modules and
  gateway routes.
* Documentation now names the gateway, registry, auth, RAG, multimodal, MCP,
  telemetry, pool jobs, feedback, drift, and deployment boundaries explicitly
  instead of deferring to core Mixle's docs.
* ``docs/_build`` is ignored for local builds.
