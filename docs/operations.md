# Operations

The 0.8 runner is a durable single-node reference with atomic JSON state and a local content store. Recover expired
leases before assigning work after restart. Workers heartbeat, checkpoint, and acknowledge cancellation using their
lease token -- `control.worker` (`run_once`/`drain`/`run_forever`) is the reference worker loop that does this,
recovering expired leases every poll before claiming and turning handler and lease failures into a typed report
instead of raising. Results and checkpoints are immutable. The deployment registry keeps current and previous
candidate IDs; health incidents may trigger one bounded rollback. Distributed queues, databases, object stores,
streaming, and SLO automation remain later work.
