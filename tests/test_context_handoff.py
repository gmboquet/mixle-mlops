from __future__ import annotations

import tempfile
import unittest

from mixle_mlops.context_handoff import ContextRunState, ContextRunStore, bundle_digest


def bundle() -> dict:
    return {
        "id": "bundle-1",
        "project_id": "science-project",
        "task": "continue solve",
        "target_kind": "model",
        "revision": 2,
        "items": [
            {
                "id": "graph",
                "kind": "artifact",
                "schema_uri": "mixle://schema/property-graph/1",
                "content_hash": "0" * 64,
                "payload": {"nodes": []},
            }
        ],
        "gaps": [{"id": "missing-lab-result", "status": "open"}],
        "required_capability_ids": ["graph.reason"],
    }


class ContextRunStoreTest(unittest.TestCase):
    def test_monitors_checkpoints_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            store = ContextRunStore(root)
            created = store.start("run-1", bundle(), model_id="model-a", idempotency_key="same-request")
            self.assertEqual(created.bundle_sha256, bundle_digest(bundle()))
            self.assertEqual(store.start("run-1", bundle(), model_id="model-a", idempotency_key="same-request").id, "run-1")

            running = store.transition("run-1", ContextRunState.RUNNING)
            self.assertEqual(running.attempt, 1)
            store.event("run-1", "need", gap_id="missing-lab-result")
            store.checkpoint("run-1", {"step": 3, "pending_actions": [{"id": "fetch-lab"}]}, state_refs=["artifact://cp"])
            failed = store.transition("run-1", ContextRunState.FAILED, error={"code": "provider_timeout"})
            self.assertEqual(failed.continuation["step"], 3)

            resumed = store.transition("run-1", ContextRunState.RUNNING)
            self.assertEqual(resumed.attempt, 2)
            completed = store.transition("run-1", ContextRunState.COMPLETED, result_refs=["knowledge://delta-1"])
            self.assertEqual(completed.result_refs, ["knowledge://delta-1"])
            self.assertEqual([event.sequence for event in completed.events], [1, 2])

    def test_rejects_identity_reuse_and_invalid_transition(self):
        with tempfile.TemporaryDirectory() as root:
            store = ContextRunStore(root)
            store.start("run", bundle(), model_id="model-a")
            with self.assertRaisesRegex(ValueError, "different bundle/model"):
                store.start("run", bundle(), model_id="model-b")
            with self.assertRaisesRegex(ValueError, "invalid context-run transition"):
                store.transition("run", ContextRunState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
