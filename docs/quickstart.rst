Quickstart
==========

This quickstart starts the MLOps gateway with the built-in echo model. It is
the smallest public path for proving the package can serve requests without
cloud credentials or a model registry.

Install for Local Gateway Work
------------------------------

``mixle-mlops`` is published on PyPI, so no clone is required for this path:

.. code-block:: console

   pip install mixle-mlops

Working from a clone of the repository instead (for example, to also run the
chat UI or change route code) uses the editable install:

.. code-block:: console

   python -m pip install -e ".[dev]"

Use additional extras only for the surface under review, such as datasets,
structured outputs, object storage, or optional cloud backends. See
:doc:`installation` for the full extras list.

Start the Gateway
-----------------

Run the FastAPI gateway:

.. code-block:: console

   python -m uvicorn mixle_mlops.gateway.app:app --host 127.0.0.1 --port 8000

The gateway always registers the dependency-free ``echo`` adapter, so this
startup path should not require a network LLM provider.

Smoke Check Health and Chat
---------------------------

In another terminal:

.. code-block:: console

   curl http://127.0.0.1:8000/health
   curl http://127.0.0.1:8000/v1/models

For an OpenAI-compatible chat smoke, post to the chat route with model
``echo``. The exact response shape is owned by the route implementation, but a
successful check should prove the gateway starts, the registry contains the
echo model, and the route returns without contacting an external provider.

Serve a Packaged Mixle Model
----------------------------

A model promoted in a Mixle production registry is served through the same
gateway, not a separate process: register it into the gateway's model
registry (see :doc:`gateway-capabilities`), then call it through
``/v1/mixle/{predict,score,latent,decide}`` like any other registered model.

Next Steps
----------

Continue with :doc:`operator-runbook` for operational boundaries and
:doc:`model-distillation-and-cross-modal-training` for training and promotion
flows.
