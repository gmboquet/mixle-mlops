Security and Data Handling
==========================

``mixle-mlops`` is the package most likely to hold credentials, account state,
deployment settings, and user payloads. Public release documentation must make
the safe operating boundary explicit.

Secrets
-------

Do not commit API keys, OAuth tokens, JWT secrets, cloud credentials, database
passwords, object-store keys, or provider credentials. Examples should use
obviously invalid sample values and environment-variable names.

Accounts and Auth
-----------------

Account, device-code, OAuth, and API-key routes should distinguish test/local
credentials from production credentials. Validation notes should say whether
auth was mocked, local, or connected to a real provider.

Storage
-------

Document which state is durable and which is cache-only. In-memory tests do not
prove Redis, object-store, or database behavior. Route docs should name the
backend assumptions when behavior depends on them.

User Payloads
-------------

Chats, files, datasets, embeddings, feedback, and generated artifacts can
contain user data. Logs and receipts should use identifiers and summaries where
possible rather than copying raw payloads.

Deployment Mutation
-------------------

Promotion and deployment should be explicit, audited state transitions. A demo
script or local feedback run should not silently mutate a production alias.
``POST /v1/models/load`` (loading a completed fine-tune into the live
registry) is gated to admins for exactly this reason: it changes what every
caller sees at ``/v1/models``, so it should not be a self-service action.

Audit Trail Expectations
------------------------

State-changing routes should leave enough evidence to answer who requested the
change, which input or artifact was used, which backend stored the result, and
how the operation can be reversed or reviewed. Local examples may use in-memory
state, but public docs should not imply that in-memory behavior proves durable
deployment safety.
