from __future__ import annotations

import json
from pathlib import Path


def test_ridge_unlearning_review_rejects_mismatched_threat_model() -> None:
    root = Path(__file__).parents[1]
    path = root / "docs/research-candidate-reviews/ridge-unlearning-20260716.json"
    review = json.loads(path.read_text())
    candidate = review["review"]

    assert review["schema_version"] == "1.0.0"
    assert review["owner_project"] == "PRJ-MLOPS"
    assert review["target_revision"] == "4f742916502c206c9478ee4c4f3e8699648c2311"
    assert review["independence"]["imports_experiment_code"] is False
    assert review["independence"]["direct_runtime_dependency"] is False
    assert candidate["bundle_digest"] == "1619b1f442126512bad6bc375a33cbe2f7381f6924bb1b4feda8c1d906688909"
    assert candidate["decision"] == "reject"
    assert candidate["threat_model"]["candidate_covers"]
    assert candidate["threat_model"]["target_requires"]
    assert any(gate["required"] and gate["result"] == "not_met" for gate in candidate["gates"])
    assert "privacy" in candidate["rationale"]
