# Operations

The 0.8 runner is a durable single-node reference with atomic JSON state and a local content store. Recover expired
leases before assigning work after restart. Workers heartbeat, checkpoint, and acknowledge cancellation using their
lease token. Results and checkpoints are immutable. The deployment registry keeps current and previous candidate IDs;
health incidents may trigger one bounded rollback. Distributed queues, databases, object stores, streaming, and SLO
automation remain later work.

The reference monitoring ledger is single-node and atomically persisted. Record observations only after resolving the
current deployment receipt, assess with a named authorization-bearing policy, and enforce only the returned persisted
unhealthy assessment id. Repeating completed enforcement returns its original receipt. If a process stops after registry
quarantine or rollback but before the monitoring receipt is written, retrying reconstructs the receipt from registry
state. Quarantine fails closed; there is intentionally no automatic unquarantine. Distributed telemetry, locks,
databases, alerts, dashboards, statistical detectors, canary traffic control, and SLO automation remain later work.
If rollback has no known non-quarantined predecessor, enforcement leaves the unhealthy candidate quarantined and
returns an explicit failure instead of continuing to serve it or inventing a target.
