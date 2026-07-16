# Requirements

- Semantic invocation identity is invariant across placement, retries, workers, and run identifiers.
- Jobs are owner-scoped, idempotent, resource-bounded, leased, cancellable, checkpointable, recoverable, and auditable.
- Artifacts are content-addressed, owner-isolated, size-bounded, and verified on every read.
- Operational completion emits `not_evaluated`; domain packages and Harness own semantic validation.
- Immutable candidates require exact trusted Factory and Harness evidence before stage/production promotion.
- Deployments keep auditable prior candidates and support explicit or health-triggered bounded rollback.
- Registry state must be independently auditable for internal consistency (receipt-log replay agreement, no dangling
  alias/previous pointer, no receipt naming an unregistered candidate, no sequence corruption) without mutating it or
  requiring the running service.
- Artifact-presence and digest verification during an audit is explicit and opt-in, never implied by a report that
  did not perform it.
