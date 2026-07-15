# Security

Never serialize credentials into jobs, artifacts, checkpoints, events, clients, or notes. Enforce owner scope before
lookup, verify content digests and sizes, reject symlink-backed state, use unguessable lease tokens and constant-time
comparison, bound events and outputs, and isolate executed workers. Production work still requires managed secrets,
encryption, database row-level isolation, supply-chain verification, retention/deletion, and an audit sink.
