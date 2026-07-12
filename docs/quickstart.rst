Quickstart
==========

This quickstart starts the MLOps gateway with the built-in echo model. It is
the smallest public path for proving the package can serve requests without
cloud credentials or a model registry.

Install for Local Gateway Work
------------------------------

From the repository root:

.. code-block:: console

   python -m pip install -e ".[docs]"

Use additional extras only for the surface under review, such as datasets,
structured outputs, object storage, or optional cloud backends.

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

The thin model-server surface in ``mixle_mlops.app`` serves a model promoted in
a Mixle production registry:

.. code-block:: console

   MIXLE_REGISTRY_ROOT=/models \
   MIXLE_MODEL_NAME=model \
   MIXLE_MODEL_ALIAS=production \
   python -m uvicorn mixle_mlops.app:app --host 127.0.0.1 --port 8001

Use this path only when a registry exists. Gateway validation and packaged
model serving are separate checks.

Next Steps
----------

Continue with :doc:`operator-runbook` for operational boundaries and
:doc:`model-distillation-and-cross-modal-training` for training and promotion
flows.
