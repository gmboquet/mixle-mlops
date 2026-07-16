# Operations

The 0.8 runner is a durable single-node reference with atomic JSON state and a local content store. Recover expired
leases before assigning work after restart. Workers heartbeat, checkpoint, and acknowledge cancellation using their
lease token. Results and checkpoints are immutable. The deployment registry keeps current and previous candidate IDs;
health incidents may trigger one bounded rollback. Distributed queues, databases, object stores, streaming, and SLO
automation remain later work.

Run `python -m mixle_mlops.control.integrity <registry-root> [--artifacts-root <path>]` after a suspected crash
mid-write, before trusting a hand-recovered `deployments.json`, or on a schedule; it exits `0` when clean and `1`
when it finds any issue, printing one line per finding. It only reads state -- a nonzero exit is a signal to
investigate and, if warranted, roll back or restore from backup, not an action the checker takes itself.
