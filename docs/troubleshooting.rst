Troubleshooting
===============

Server Route Fails Locally
--------------------------

Record the start command, backend configuration, and environment variables
used. Confirm whether the route depends on optional extras such as storage,
datasets, export, local models, or MCP.

Import Error For Core ``mixle``
-------------------------------

Install core ``mixle`` into the environment or set ``PYTHONPATH`` to the local
core checkout during validation. Do not hide this dependency by relying on an
accidental shell state.

Storage Or Cache Behavior Differs
---------------------------------

Check which backend is configured. In-memory, SQLite, Redis, object stores, and
managed databases have different durability and concurrency behavior. Docs and
tests should say which one was used.

Drift Or Feedback State Looks Wrong
-----------------------------------

Separate collection from promotion. A metric can be observed, a candidate can
be created, a reviewer can approve it, and serving state can later change.
Debug each state transition independently.

Docs Build Fails
----------------

Run:

.. code-block:: console

   python -m sphinx -W -b html docs docs/_build/html
