mixle-mlops
===========

``mixle-mlops`` is the serving, gateway, registry, feedback, storage, and
deployment package for the Mixle ecosystem. It hosts Mixle models and
OpenAI-compatible model providers behind one application surface, with account
management, API keys, RAG, multimodal input, MCP tooling, feedback loops,
dataset export, and deployment helpers.

This manual also covers the gateway capability surfaces: substrate serving,
telemetry, pool jobs, spend rails, creation verbs, lineage, and drift
retraining.

Start Here
----------

Start with :doc:`quickstart` to run the local gateway with the built-in echo
model. Use :doc:`operator-runbook` for operational boundaries and
:doc:`gateway-operating-contract` for route behavior. Use
:doc:`model-distillation-and-cross-modal-training` for training, registry, and
promotion workflows.

.. toctree::
   :caption: Start Here
   :hidden:
   :maxdepth: 2

   installation
   quickstart
   overview
   package-map
   operator-runbook
   gateway-operating-contract
   model-distillation-and-cross-modal-training
   release-notes
   changelog
   security-and-data
   example-execution-manifest
   gateway-capabilities
   api-overview
   validation
   troubleshooting

.. toctree::
   :caption: Reference
   :hidden:
   :maxdepth: 2

   api/modules

Review Standard
---------------

Treat this manual as both operator documentation and release evidence. A route,
provider adapter, registry action, dataset export, feedback loop, or training
path should not be considered public until the docs name its configuration,
authentication behavior, persistence expectations, validation command, and
failure mode. Gateway features can affect user data and served model behavior,
so undocumented defaults are release risks rather than implementation details.

Reader Path
-----------

Operators should begin with installation, quickstart, and the operator runbook.
Reviewers should pair the gateway operating contract with validation and API
reference pages. Contributors changing training or promotion should also read
the distillation and cross-modal training guide before editing route behavior.
