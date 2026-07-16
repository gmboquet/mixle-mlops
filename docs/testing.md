# Testing

Focused tests cover semantic idempotency, owner isolation, invalid transitions, duplicate delivery, lease expiry,
heartbeat/retry, cancellation races, checkpoint tampering, restart/recovery, output bounds, forged or failed evidence,
untrusted authorities, missing suites, immutable candidate collisions, promotion, forced failure, rollback, and the
worker harness's claim-to-resolution loop (retryable-then-successful attempts, terminal non-retryable failure, an
unclassified handler exception failing closed, an oversized result rejected without raising, cooperative cancellation
through the lease token, pre-claim lease recovery, and a lease lost mid-handler staying recoverable). Later gates add
database migrations, auth/key rotation, multi-worker queues, streaming, quotas, backups, and deployments.

The focused deployment-monitoring tests cover exact deployment binding, immutable observation ids, finite metrics,
timezone-bearing timestamps, bounded healthy, insufficient, and unhealthy windows, missing metrics as breaches, policy
validation, restart, stale-assessment rejection, quarantine-only fail-closed behavior, quarantine-aware promotion and
rollback, idempotent enforcement, and recovery after an interrupted registry transition.

The focused integrity tests cover a clean promotion/rollback history replaying without issues, a dangling alias or
previous pointer naming an unregistered candidate, a receipt naming an unregistered candidate or carrying an
unrecognized action, a corrupted receipt sequence (gap and duplicate together), live state disagreeing with a
receipt-log replay while every candidate stays individually valid, and both a missing and a digest-mismatched
candidate artifact when an artifact store is supplied.
