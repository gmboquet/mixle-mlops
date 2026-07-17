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

The focused pde-surrogate-bridge tests (skipped, not failed, when `mixle_pde` or `numpy` are not installed) cover:
training a real `LinearOperatorSurrogate` end to end through the durable worker (claim, a mid-fit checkpoint,
calibrate, complete) with the unpickled surrogate genuinely predicting; the trained result passing
`check_registry_integrity` once registered; a surrogate landed from pde's own `ArtifactStore` instead, with its
lineage (parent snapshot digest, pde-side metadata) carried into the candidate and both stores agreeing on the same
sha256 digest; the full train -> register -> integrity-check -> promote -> resolve lifecycle in one pass; and an
`imprecise` calibration producing a failing HARNESS receipt that `registry.promote` genuinely refuses.
