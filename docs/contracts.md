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
