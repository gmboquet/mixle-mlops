# Architecture

`control.contracts` defines owner, capability, artifact, invocation, resource, retry, job, candidate, evidence, and
promotion contracts. `LocalArtifactStore` is the owner-scoped content-addressed reference store. `DurableLocalRunner`
persists jobs atomically and coordinates leases, events, checkpoints, cancellation, retry, and recovery while workers
execute injected domain code. `control.worker` is that worker: `run_once`/`drain`/`run_forever` claim a job, invoke an
injected domain-neutral handler through a `WorkerContext` (heartbeat/progress/checkpoint), and resolve it through the
runner's own state machine, turning handler exceptions and lease races into a typed `WorkReport` instead of raising.
`DeploymentRegistry` stores immutable candidates and promotion/rollback history. Existing gateway, account, storage,
compute, provider, and frontend modules become adapters over this control boundary.

`DeploymentMonitor` is a separate durable local ledger. Collectors submit finite metrics bound to one alias, candidate,
and deployment receipt. A versioned policy evaluates only a bounded exact window, persists insufficient, healthy, or
unhealthy state, and names the external authorization governing quarantine or rollback. Enforcement reloads the
persisted assessment, rechecks live deployment identity, quarantines the unhealthy candidate, and applies at most one
rollback. Registry receipts let an interrupted enforcement be reconstructed without repeating the transition.

`check_registry_integrity` (`control.integrity`) is a stateless, read-only auditor over a loaded `DeploymentRegistry`.
It replays the receipt log to re-derive the alias/previous projection, cross-checks every alias and previous pointer
against registered candidates, and -- when handed a `LocalArtifactStore` -- reuses that store's own digest
verification to confirm each candidate's artifact is still present and byte-exact. It holds no state of its own and
never writes to the registry it inspects.

`control.pde_surrogate` is the MP-N8 bridge from a trained `mixle_pde` operator surrogate into this boundary:
`train_operator_surrogate_job` is a `WorkerHandler` that fits and calibrates a real `LinearOperatorSurrogate` from a
job's own parameters, so training runs on the existing durable-job machinery instead of a bespoke script;
`land_pde_artifact` re-lands an artifact already in pde's own content-addressed `ArtifactStore` into this platform's
`LocalArtifactStore`, asserting the two independently-implemented stores agree on the same sha256 digest;
`register_pde_operator_surrogate` turns either path's landed artifact into a `ModelCandidate` with FACTORY/HARNESS
evidence derived from the surrogate's own held-out calibration report, so `check_registry_integrity` and
`registry.promote`/`rollback` govern it exactly like any other candidate. `mixle_pde` remains an optional,
lazily-imported dependency, matching `mcp/physics_tools.py`'s precedent for this cross-repo boundary.
