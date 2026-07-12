Troubleshooting
===============

Server Route Fails Locally
--------------------------

Record the start command, backend configuration, and environment variables
used. Confirm whether the route depends on optional extras such as storage,
datasets, export, local models, or MCP.

Also capture the response body and status code. A startup error, validation
error, auth failure, provider failure, and storage failure require different
owners and should not be collapsed into a generic gateway issue.

Import Error for Core ``mixle``
-------------------------------

Install core ``mixle`` into the environment or set ``PYTHONPATH`` to the local
core checkout during validation. Do not hide this dependency by relying on an
accidental shell state.

Storage or Cache Behavior Differs
---------------------------------

Check which backend is configured. In-memory, SQLite, Redis, object stores, and
managed databases have different durability and concurrency behavior. Docs and
tests should say which one was used.

Drift or Feedback State Looks Wrong
-----------------------------------

Separate collection from promotion. A metric can be observed, a candidate can
be created, a reviewer can approve it, and serving state can later change.
Debug each state transition independently.

Docs Build Fails
----------------

Run:

.. code-block:: console

   make -C docs html SPHINXOPTS="-W --keep-going"

If autodoc fails, check optional imports first. The public API reference covers
routes and helper modules that may need extras for documents, datasets, export,
MCP, local models, or image generation. Missing optional dependencies should
produce clear skip or install guidance, not an unexplained import crash.

Promotion State Changes Unexpectedly
------------------------------------

Check whether the change came from feedback collection, candidate creation,
review approval, deployment, or manual registry mutation. Preserve the receipt
or API response for the state transition being debugged. Direct alias edits are
not release evidence for the reviewed promotion path.

Triage Order
------------

For operational failures, check configuration first, then authentication, then
provider availability, then persistence state, then route behavior. This order
keeps missing credentials or optional extras from being mistaken for model or
registry regressions.
