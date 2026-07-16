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
