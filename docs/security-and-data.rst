Security And Data Handling
==========================

``mixle-mlops`` is the package most likely to hold credentials, account state,
deployment settings, and user payloads. Public release documentation must make
the safe operating boundary explicit.

Secrets
-------

Do not commit API keys, OAuth tokens, JWT secrets, cloud credentials, database
passwords, object-store keys, or provider credentials. Examples should use
placeholder values and environment-variable names.

Accounts And Auth
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

Release Checklist
-----------------

Before release:

* inspect examples and docs for real secrets;
* record backend configuration for route smoke tests;
* verify auth and deployment paths have clear local/test modes;
* preserve observed/candidate/approved/deployed distinctions in drift flows;
* build docs with warnings as errors.
