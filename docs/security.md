# Security

Never serialize credentials into jobs, artifacts, checkpoints, events, clients, or notes. Enforce owner scope before
lookup, verify content digests and sizes, reject symlink-backed state, use unguessable lease tokens and constant-time
comparison, bound events and outputs, and isolate executed workers. Production work still requires managed secrets,
encryption, database row-level isolation, supply-chain verification, retention/deletion, and an audit sink.

Monitoring rejects non-finite values, naive timestamps, ambiguous policies, duplicate threshold or observation
identity, stale deployment bindings, unknown or non-unhealthy assessments, quarantined rollback targets, and attempts to
re-promote quarantined candidates. A policy stores an authorization identity, but this local reference does not verify
its issuer, signature, expiry, or revocation; governed adoption must do that before installing the policy. The local JSON
ledgers use atomic replacement but do not provide multi-process locking or tamper signatures.
