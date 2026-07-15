# Contracts

Schema `1.0.0` uses deterministic JSON and SHA-256 semantic identity. `InvocationSpec.semantic_id` excludes job, worker,
lease, attempt, and placement. Artifact refs include owner scope, digest, exact size, media type, URI, and semantic type.
Job state is queued, leased, running, succeeded, failed, or cancelled; state changes require valid unexpired leases.
Candidate promotion receipts name the exact artifact digest and authorities.
