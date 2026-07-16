# Contracts

Schema `1.0.0` uses deterministic JSON and SHA-256 semantic identity. `InvocationSpec.semantic_id` excludes job, worker,
lease, attempt, and placement. Artifact refs include owner scope, digest, exact size, media type, URI, and semantic type.
Job state is queued, leased, running, succeeded, failed, or cancelled; state changes require valid unexpired leases.
Candidate promotion receipts name the exact artifact digest and authorities.

`check_registry_integrity` produces an `IntegrityReport` (`checked_candidates`, `checked_receipts`, `checked_aliases`,
`checked_artifacts`, `issues`) -- a derived, in-memory value, not part of the durable schema `1.0.0` payload.
`checked_artifacts` is `None`, not zero, when no artifact store was supplied, so a clean report never implies
artifacts were verified unless it also says how many. Each `IntegrityIssue` names a typed `IntegrityFinding` (dangling
alias/previous, unknown receipt candidate or action, sequence gap/duplicate, projection drift, missing or
digest-mismatched artifact), a subject, and a human-readable detail.
