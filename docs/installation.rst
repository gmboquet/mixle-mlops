Installation
============

``mixle-mlops`` has a larger dependency surface than the modeling packages
because it owns gateway, account, storage, provider, dataset, document, cloud,
and MCP integrations. Install only the extras required for the surface you are
validating, and record those extras in release evidence.

PyPI Install
------------

``mixle-mlops`` is published on PyPI. For running the gateway without cloning
the repository:

.. code-block:: console

   pip install mixle-mlops
   pip install "mixle-mlops[all]"   # + documents, datasets, export, structured, local, cloud, mcp, image

This installs the gateway package only. The chat UI (``frontend/``) is a
separate Next.js application that is not shipped on PyPI; clone the repository
for that surface (see below).

Local Development Install
--------------------------

For contributing to the package itself, clone the repository and install from
source:

.. code-block:: console

   python -m pip install -e ".[dev]"

Install optional surfaces as needed:

.. code-block:: console

   python -m pip install -e ".[documents,scale,export,datasets,structured,local,cloud,mcp,image]"
   python -m pip install -e ".[docs]"

Run the gateway locally:

.. code-block:: console

   mixle-serve

Run the chat UI:

.. code-block:: console

   cd frontend
   npm install
   npm run dev

Docker compose smoke run:

.. code-block:: console

   cp deploy/.env.example deploy/.env
   docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d gateway ollama
   curl localhost:8000/v1/models

Build this documentation:

.. code-block:: console

   make -C docs html SPHINXOPTS="-W --keep-going"

Common Install Profiles
-----------------------

``base``
    Gateway application, local echo model, model registry, and core API
    routes.

``documents`` / ``datasets`` / ``export``
    Document parsing, generated datasets, and artifact export workflows.

``local`` / ``cloud``
    Local model providers, cloud launch helpers, and deployment utilities.

``mcp``
    MCP client/server integration and tool-schema bridging.

Dependency Boundaries
---------------------

Keep optional provider dependencies isolated. A local account, storage, or
gateway smoke test should not require cloud credentials, rented-GPU packages, or
document-processing libraries unless that surface is being validated directly.
This separation keeps release checks fast and makes missing optional extras
obvious to operators.

Validation After Install
------------------------

After installing the selected extras, run a local gateway smoke check and a
strict docs build:

.. code-block:: console

   mixle-serve --help
   make -C docs html SPHINXOPTS="-W --keep-going"

If a provider, object store, Redis, or cloud dependency is absent, the related
feature should skip explicitly or report a clear configuration error rather
than fail on an unrelated import.
