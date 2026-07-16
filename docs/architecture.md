# Architecture

`control.contracts` defines owner, capability, artifact, invocation, resource, retry, job, candidate, evidence, and
promotion contracts. `LocalArtifactStore` is the owner-scoped content-addressed reference store. `DurableLocalRunner`
persists jobs atomically and coordinates leases, events, checkpoints, cancellation, retry, and recovery while workers
execute injected domain code. `DeploymentRegistry` stores immutable candidates and promotion/rollback history. Existing
gateway, account, storage, compute, provider, and frontend modules become adapters over this control boundary.

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
