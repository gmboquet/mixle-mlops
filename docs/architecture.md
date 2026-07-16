# Architecture

`control.contracts` defines owner, capability, artifact, invocation, resource, retry, job, candidate, evidence, and
promotion contracts. `LocalArtifactStore` is the owner-scoped content-addressed reference store. `DurableLocalRunner`
persists jobs atomically and coordinates leases, events, checkpoints, cancellation, retry, and recovery while workers
execute injected domain code. `control.worker` is that worker: `run_once`/`drain`/`run_forever` claim a job, invoke an
injected domain-neutral handler through a `WorkerContext` (heartbeat/progress/checkpoint), and resolve it through the
runner's own state machine, turning handler exceptions and lease races into a typed `WorkReport` instead of raising.
`DeploymentRegistry` stores immutable candidates and promotion/rollback history. Existing gateway, account, storage,
compute, provider, and frontend modules become adapters over this control boundary.
