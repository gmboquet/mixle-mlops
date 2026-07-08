mixle-mlops
===========

``mixle-mlops`` is the serving, gateway, registry, feedback, storage, and
deployment package for the Mixle ecosystem. It hosts Mixle models and
OpenAI-compatible model providers behind one application surface, with account
management, API keys, RAG, multimodal input, MCP tooling, feedback loops,
dataset export, and deployment helpers.

For 0.6.3, the package also documents the gateway capability work for substrate
serving, telemetry, pool jobs, spend rails, creation verbs, lineage, and drift
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
   README
   release-0-6-3
   changelog
   security-and-data
   example-execution-manifest
   0.6.3-gateway-capabilities
   api-overview
   validation
   troubleshooting

.. toctree::
   :caption: Reference
   :hidden:
   :maxdepth: 2

   api/modules
