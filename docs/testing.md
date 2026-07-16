# Testing

Focused tests cover semantic idempotency, owner isolation, invalid transitions, duplicate delivery, lease expiry,
heartbeat/retry, cancellation races, checkpoint tampering, restart/recovery, output bounds, forged or failed evidence,
untrusted authorities, missing suites, immutable candidate collisions, promotion, forced failure, rollback, and the
worker harness's claim-to-resolution loop (retryable-then-successful attempts, terminal non-retryable failure, an
unclassified handler exception failing closed, an oversized result rejected without raising, cooperative cancellation
through the lease token, pre-claim lease recovery, and a lease lost mid-handler staying recoverable). Later gates add
database migrations, auth/key rotation, multi-worker queues, streaming, quotas, backups, and deployments.
