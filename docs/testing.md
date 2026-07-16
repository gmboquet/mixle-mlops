# Testing

Focused tests cover semantic idempotency, owner isolation, invalid transitions, duplicate delivery, lease expiry,
heartbeat/retry, cancellation races, checkpoint tampering, restart/recovery, output bounds, forged or failed evidence,
untrusted authorities, missing suites, immutable candidate collisions, promotion, forced failure, and rollback. Later
gates add database migrations, auth/key rotation, multi-worker queues, streaming, quotas, backups, and deployments.

The focused integrity tests cover a clean promotion/rollback history replaying without issues, a dangling alias or
previous pointer naming an unregistered candidate, a receipt naming an unregistered candidate or carrying an
unrecognized action, a corrupted receipt sequence (gap and duplicate together), live state disagreeing with a
receipt-log replay while every candidate stays individually valid, and both a missing and a digest-mismatched
candidate artifact when an artifact store is supplied.
