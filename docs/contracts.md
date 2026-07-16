# Contracts

## Governed capability adoption

`GovernedDeploymentRegistry` extends the existing evidence-gated deployment
registry. Adoption additionally requires a verified independent acceptance
attestation, a Core-compatible scoped authorization decision that is effective
and not revoked or expired, and a trusted architecture epoch matching the
candidate's build metadata. These gates remain separate records; an accepted
evaluation cannot authorize itself, and a job or build success cannot bypass
any gate.

Schema `1.0.0` uses deterministic JSON and SHA-256 semantic identity. `InvocationSpec.semantic_id` excludes job, worker,
lease, attempt, and placement. Artifact refs include owner scope, digest, exact size, media type, URI, and semantic type.
Job state is queued, leased, running, succeeded, failed, or cancelled; state changes require valid unexpired leases.
Candidate promotion receipts name the exact artifact digest and authorities.

Monitoring schema `1.0.0` defines `MetricThreshold`, `MonitoringPolicy`, `HealthObservation`, `HealthAssessment`, and
`EnforcementReceipt`. Policies carry a stable id/version, semantic SHA-256, finite maximum/minimum thresholds, bounded
window and minimum sample counts, breach tolerance, allowed aliases, action, and external authorization identity.
Observations bind metrics to the exact alias, candidate, and deployment sequence. Assessments retain their exact ordered
observation ids and hashes, the complete policy, per-metric missing and breach counts, state, and action. Missing required
metrics count as breaches; insufficient evidence is never reported as healthy. Quarantine and rollback retain incident, policy,
authorization, actor, unhealthy candidate, replacement candidate, and registry receipt identity.

`check_registry_integrity` produces an `IntegrityReport` (`checked_candidates`, `checked_receipts`, `checked_aliases`,
`checked_artifacts`, `issues`) -- a derived, in-memory value, not part of the durable schema `1.0.0` payload.
`checked_artifacts` is `None`, not zero, when no artifact store was supplied, so a clean report never implies
artifacts were verified unless it also says how many. Each `IntegrityIssue` names a typed `IntegrityFinding` (dangling
alias/previous, unknown receipt candidate or action, sequence gap/duplicate, projection drift, missing or
digest-mismatched artifact), a subject, and a human-readable detail.
