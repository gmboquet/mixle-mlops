Validation
==========

The package has a broad test suite spanning gateway routes, providers,
feedback, datasets, RAG, accounts, object storage, MCP, multimodal payloads,
structured output, task cascades, pool jobs, substrate serving, telemetry,
creation verbs, and drift retraining.

Focused validation:

.. code-block:: console

   PYTHONPATH=../mixle python -m pytest \
       tests/test_pool_serving.py \
       tests/test_substrate_serving.py \
       tests/test_telemetry_serving.py \
       tests/test_verbs_serving.py \
       tests/test_drift_retrain.py

The local workspace needs the core Mixle package on ``PYTHONPATH`` unless
``mixle`` is installed into the active environment.

Choose focused tests by surface. Account, provider, object-storage, dataset,
document, and gateway changes should each include the smallest test set that
exercises the changed contract plus one smoke path through the public API when
the behavior is user-facing.

Run the full suite from the package root with:

.. code-block:: console

   python -m pytest

Operational Smoke Checks
------------------------

For route or service changes, pair unit tests with a small local smoke check:
start the gateway, exercise the changed route, and confirm the response shape
against the backend actually configured (in-memory, SQLite, Redis, or object
store).

Strict Docs Gate
----------------

For release review, build with warnings treated as errors:

.. code-block:: console

   make -C docs html SPHINXOPTS="-W --keep-going"

Clean-Archive Documentation Gate
--------------------------------

Before public release, also build the docs from tracked files only:

.. code-block:: console

   tmp=$(mktemp -d)
   git archive HEAD | tar -x -C "$tmp"
   PYTHONPATH="$tmp:${MIXLE_CORE_CHECKOUT:?set MIXLE_CORE_CHECKOUT to a core mixle checkout}" \
     make -C "$tmp/docs" html SPHINXOPTS="-W --keep-going"

Use an installed core ``mixle`` package instead of the workspace path when
validating published artifacts. This gate catches autodoc failures hidden by a
dirty working tree.

Evidence Expectations
---------------------

Validation notes should name the optional extras installed, the backend used for
stateful services, and whether external credentials were real, local-only, or
mocked. For failures, keep the command and configuration with the result so the
owner can distinguish dependency setup problems from behavior regressions.
