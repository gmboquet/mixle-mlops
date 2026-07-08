Validation
==========

The package has a broad test suite spanning gateway routes, providers,
feedback, datasets, RAG, accounts, object storage, MCP, multimodal payloads,
structured output, task cascades, pool jobs, substrate serving, telemetry,
creation verbs, and drift retraining.

Focused 0.6.3 validation:

.. code-block:: console

   PYTHONPATH=/Users/grantboquet/mixle/mixle python -m pytest \
       tests/test_pool_serving.py \
       tests/test_substrate_serving.py \
       tests/test_telemetry_serving.py \
       tests/test_verbs_serving.py \
       tests/test_drift_retrain.py

The local workspace needs the core Mixle package on ``PYTHONPATH`` unless
``mixle`` is installed into the active environment.

Run the full suite from the package root with:

.. code-block:: console

   python -m pytest
