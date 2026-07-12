Release Notes
=============

``mixle-mlops`` hosts Mixle models and OpenAI-compatible LLMs behind one
gateway, with accounts and API keys, RAG and multimodal input, an MCP server,
feedback and self-evolution loops, dataset export, and deployment helpers for
local and cloud environments.

0.6.3
-----

0.6.3 is a documentation-completion release: the gateway, registry, training,
and deployment surfaces built in earlier commits are now documented well
enough to review. The manual covers the route-by-route operating contract
(:doc:`gateway-operating-contract`), the substrate serving, telemetry, pool
job, and creation-verb surfaces (:doc:`gateway-capabilities`), and
distillation and cross-modal training
(:doc:`model-distillation-and-cross-modal-training`), backed by a generated
API reference for every public module and route.

Known Risks
-----------

Public release still needs:

* a clean package build and install from a fresh environment;
* route and service tests run together as a single release gate, not only
  per-surface (see :doc:`validation` for the current focused test commands);
* a recorded mock/skip/real status for every external service a route
  depends on;
* verification that this package still co-installs cleanly with the rest of
  the Mixle family.
