# Security

Never serialize credentials into jobs, artifacts, checkpoints, events, clients, or notes. Enforce owner scope before
lookup, verify content digests and sizes, reject symlink-backed state, use unguessable lease tokens and constant-time
comparison, bound events and outputs, and isolate executed workers. Production work still requires managed secrets,
encryption, database row-level isolation, supply-chain verification, retention/deletion, and an audit sink.

The integrity checker only reads `deployments.json` and, optionally, artifact bytes for digest verification; it never
writes to either and needs no access beyond what reading the registry already requires. Findings are returned as
data, not raised as exceptions, so a corrupted registry cannot be mistaken for a crashed check -- but the checker
itself does not repair, quarantine, or roll back anything it finds wrong.
