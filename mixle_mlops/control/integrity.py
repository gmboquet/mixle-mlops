"""Read-only integrity auditing for the durable deployment registry.

`DeploymentRegistry` keeps two things on disk: an append-only receipt log (the audit trail of every
promotion and rollback) and a derived, mutable projection (`aliases`/`previous`) that `promote`/`rollback`
maintain in lockstep with that log. Nothing today re-checks that the two still agree once the file is back
on disk -- a crash between writes, a hand-edited `deployments.json`, or a future bug could let them drift
apart silently, and neither a dangling alias nor a receipt naming an unregistered candidate would be caught
by any code path that only ever calls `resolve()`/`candidate()` for one alias at a time.

`check_registry_integrity` re-derives the projection by replaying the receipt log in sequence order and
compares it against the live state, checks every alias and "previous" pointer resolves to a registered
candidate, checks the receipt sequence itself has no gaps or duplicates, and -- only when an artifact store
is supplied -- verifies every candidate's artifact is still present and byte-exact. It never mutates the
registry and never judges model quality; it only judges whether the registry's own bookkeeping is
self-consistent. Whether an artifact store was even asked to verify is preserved in the report rather than
folded into a boolean, so a clean report never silently claims artifacts were checked when they were not.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import StrEnum

from .artifacts import LocalArtifactStore
from .contracts import ModelCandidate, OperationalError
from .registry import DeploymentReceipt, DeploymentRegistry

_KNOWN_ACTIONS = ("promote", "rollback")


class IntegrityFinding(StrEnum):
    DANGLING_ALIAS = "dangling_alias"
    DANGLING_PREVIOUS = "dangling_previous"
    RECEIPT_UNKNOWN_CANDIDATE = "receipt_unknown_candidate"
    RECEIPT_UNKNOWN_ACTION = "receipt_unknown_action"
    RECEIPT_SEQUENCE_GAP = "receipt_sequence_gap"
    RECEIPT_SEQUENCE_DUPLICATE = "receipt_sequence_duplicate"
    PROJECTION_DRIFT = "projection_drift"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"


@dataclass(frozen=True)
class IntegrityIssue:
    finding: IntegrityFinding
    subject: str
    detail: str


@dataclass(frozen=True)
class IntegrityReport:
    checked_candidates: int
    checked_receipts: int
    checked_aliases: int
    checked_artifacts: int | None
    issues: tuple[IntegrityIssue, ...] = ()

    @property
    def clean(self) -> bool:
        return len(self.issues) == 0

    def issues_of(self, finding: IntegrityFinding) -> tuple[IntegrityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.finding is finding)


def check_registry_integrity(
    registry: DeploymentRegistry,
    *,
    artifacts: LocalArtifactStore | None = None,
) -> IntegrityReport:
    """Audit one loaded `DeploymentRegistry` and return a typed report; never raises on a dirty registry."""
    candidates = registry.candidates
    receipts = registry.receipts
    aliases = registry.aliases
    previous = registry.previous
    known_candidate_ids = {candidate.id for candidate in candidates}

    issues: list[IntegrityIssue] = []
    issues.extend(_sequence_issues(receipts))
    issues.extend(_unknown_candidate_issues(receipts, known_candidate_ids))
    issues.extend(_projection_drift_issues(receipts, live_aliases=aliases, live_previous=previous))
    issues.extend(_dangling_pointer_issues(aliases, previous, known_candidate_ids))

    checked_artifacts: int | None = None
    if artifacts is not None:
        checked_artifacts = len(candidates)
        issues.extend(_artifact_issues(candidates, artifacts))

    return IntegrityReport(
        checked_candidates=len(candidates),
        checked_receipts=len(receipts),
        checked_aliases=len(aliases),
        checked_artifacts=checked_artifacts,
        issues=tuple(issues),
    )


def _sequence_issues(receipts: tuple[DeploymentReceipt, ...]) -> list[IntegrityIssue]:
    counts: dict[int, int] = {}
    for receipt in receipts:
        counts[receipt.sequence] = counts.get(receipt.sequence, 0) + 1
    issues: list[IntegrityIssue] = []
    for sequence, count in sorted(counts.items()):
        if count > 1:
            issues.append(
                IntegrityIssue(
                    finding=IntegrityFinding.RECEIPT_SEQUENCE_DUPLICATE,
                    subject=str(sequence),
                    detail=f"sequence {sequence} appears {count} times across {len(receipts)} receipts",
                )
            )
    for expected in range(len(receipts)):
        if expected not in counts:
            issues.append(
                IntegrityIssue(
                    finding=IntegrityFinding.RECEIPT_SEQUENCE_GAP,
                    subject=str(expected),
                    detail=f"sequence {expected} is missing from {len(receipts)} receipts",
                )
            )
    return issues


def _unknown_candidate_issues(
    receipts: tuple[DeploymentReceipt, ...],
    known_candidate_ids: set[str],
) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    for receipt in receipts:
        if receipt.action not in _KNOWN_ACTIONS:
            issues.append(
                IntegrityIssue(
                    finding=IntegrityFinding.RECEIPT_UNKNOWN_ACTION,
                    subject=f"receipt[{receipt.sequence}]",
                    detail=f"receipt {receipt.sequence} has an unrecognized action {receipt.action!r}",
                )
            )
        for role, candidate_id in (
            ("candidate_id", receipt.candidate_id),
            ("previous_candidate_id", receipt.previous_candidate_id),
        ):
            if candidate_id is not None and candidate_id not in known_candidate_ids:
                issues.append(
                    IntegrityIssue(
                        finding=IntegrityFinding.RECEIPT_UNKNOWN_CANDIDATE,
                        subject=f"receipt[{receipt.sequence}].{role}",
                        detail=(
                            f"receipt {receipt.sequence} ({receipt.action} {receipt.alias!r}) names "
                            f"unregistered candidate {candidate_id!r}"
                        ),
                    )
                )
    return issues


def _replay(receipts: tuple[DeploymentReceipt, ...]) -> tuple[dict[str, str], dict[str, str]]:
    aliases: dict[str, str] = {}
    previous: dict[str, str] = {}
    for receipt in sorted(receipts, key=lambda item: item.sequence):
        if receipt.action == "promote":
            if receipt.previous_candidate_id is not None:
                previous[receipt.alias] = receipt.previous_candidate_id
            if receipt.candidate_id is not None:
                aliases[receipt.alias] = receipt.candidate_id
        elif receipt.action == "rollback":
            if receipt.candidate_id is not None:
                aliases[receipt.alias] = receipt.candidate_id
            if receipt.previous_candidate_id is not None:
                previous[receipt.alias] = receipt.previous_candidate_id
        # An unrecognized action is already reported by `_unknown_candidate_issues`; replay does not
        # guess at its effect, so the projection may legitimately keep drifting from that point on.
    return aliases, previous


def _projection_drift_issues(
    receipts: tuple[DeploymentReceipt, ...],
    *,
    live_aliases: dict[str, str],
    live_previous: dict[str, str],
) -> list[IntegrityIssue]:
    replayed_aliases, replayed_previous = _replay(receipts)
    issues: list[IntegrityIssue] = []
    for label, replayed, live in (
        ("aliases", replayed_aliases, live_aliases),
        ("previous", replayed_previous, live_previous),
    ):
        for key in sorted(set(replayed) | set(live)):
            if replayed.get(key) != live.get(key):
                issues.append(
                    IntegrityIssue(
                        finding=IntegrityFinding.PROJECTION_DRIFT,
                        subject=f"{label}[{key}]",
                        detail=(
                            f"replaying the receipt log gives {label}[{key!r}] = {replayed.get(key)!r}, "
                            f"but the live registry state has {live.get(key)!r}"
                        ),
                    )
                )
    return issues


def _dangling_pointer_issues(
    aliases: dict[str, str],
    previous: dict[str, str],
    known_candidate_ids: set[str],
) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    for alias, candidate_id in sorted(aliases.items()):
        if candidate_id not in known_candidate_ids:
            issues.append(
                IntegrityIssue(
                    finding=IntegrityFinding.DANGLING_ALIAS,
                    subject=alias,
                    detail=f"alias {alias!r} points at unregistered candidate {candidate_id!r}",
                )
            )
    for alias, candidate_id in sorted(previous.items()):
        if candidate_id not in known_candidate_ids:
            issues.append(
                IntegrityIssue(
                    finding=IntegrityFinding.DANGLING_PREVIOUS,
                    subject=alias,
                    detail=f"previous[{alias!r}] points at unregistered candidate {candidate_id!r}",
                )
            )
    return issues


def _artifact_issues(
    candidates: tuple[ModelCandidate, ...],
    artifacts: LocalArtifactStore,
) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    for candidate in candidates:
        try:
            artifacts.get(candidate.artifact.owner, candidate.artifact)
        except FileNotFoundError:
            issues.append(
                IntegrityIssue(
                    finding=IntegrityFinding.ARTIFACT_MISSING,
                    subject=candidate.id,
                    detail=(
                        f"candidate {candidate.id!r} artifact {candidate.artifact.sha256} is not present "
                        "in the artifact store"
                    ),
                )
            )
        except OperationalError as error:
            issues.append(
                IntegrityIssue(
                    finding=IntegrityFinding.ARTIFACT_DIGEST_MISMATCH,
                    subject=candidate.id,
                    detail=f"candidate {candidate.id!r} artifact failed verification: {error}",
                )
            )
    return issues


def _render(report: IntegrityReport) -> str:
    lines = [
        f"candidates={report.checked_candidates} receipts={report.checked_receipts} "
        f"aliases={report.checked_aliases} artifacts_checked={report.checked_artifacts!r}"
    ]
    if report.clean:
        lines.append("registry is internally consistent")
    else:
        for issue in report.issues:
            lines.append(f"[{issue.finding.value}] {issue.subject}: {issue.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a durable deployment registry for internal consistency.")
    parser.add_argument("registry_root", help="root directory passed to DeploymentRegistry")
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="optional LocalArtifactStore root to verify candidate artifacts against",
    )
    arguments = parser.parse_args(argv)

    registry = DeploymentRegistry(arguments.registry_root)
    artifacts = LocalArtifactStore(arguments.artifacts_root) if arguments.artifacts_root else None
    report = check_registry_integrity(registry, artifacts=artifacts)
    print(_render(report))
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "IntegrityFinding",
    "IntegrityIssue",
    "IntegrityReport",
    "check_registry_integrity",
]
