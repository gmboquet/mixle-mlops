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

Operational Smoke Checks
------------------------

For route or service changes, pair unit tests with a small local smoke check.
Record:

* server start command;
* route exercised;
* backend configuration, such as in-memory, SQLite, Redis, or object store;
* whether credentials were fake/local or real;
* expected response shape;
* logs or artifacts produced by the check.

Strict Docs Gate
----------------

For release review, build with warnings treated as errors:

.. code-block:: console

   python -m sphinx -W -b html docs docs/_build/html
