# Testing

Focused tests cover semantic idempotency, owner isolation, invalid transitions, duplicate delivery, lease expiry,
heartbeat/retry, cancellation races, checkpoint tampering, restart/recovery, output bounds, forged or failed evidence,
untrusted authorities, missing suites, immutable candidate collisions, promotion, forced failure, and rollback. Later
gates add database migrations, auth/key rotation, multi-worker queues, streaming, quotas, backups, and deployments.

The focused deployment-monitoring tests cover exact deployment binding, immutable observation ids, finite metrics,
timezone-bearing timestamps, bounded healthy, insufficient, and unhealthy windows, missing metrics as breaches, policy
validation, restart, stale-assessment rejection, quarantine-only fail-closed behavior, quarantine-aware promotion and
rollback, idempotent enforcement, and recovery after an interrupted registry transition.
