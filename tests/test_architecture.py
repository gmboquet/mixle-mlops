from mixle_mlops import (
    ArchitectureTransitionController,
    ConsumerMigration,
    DualRunPolicy,
    MigrationPlan,
    ProviderContract,
    RollbackPlan,
    TransitionAuthorization,
    TransitionState,
)


def plan(*, identifier="plan-1", epoch="epoch-7", targets=("provider",)) -> MigrationPlan:
    return MigrationPlan(
        identifier,
        "proposal-1",
        epoch,
        (ProviderContract("provider-v2", "solver", "2", "input/v1", "output/v1"),),
        (ConsumerMigration("consumer-1", "provider-v1", "provider-v2", "dual_read"),),
        DualRunPolicy(("correctness", "latency"), 10, 2),
        RollbackPlan(("correctness_regression",), ("provider-v1",)),
        targets,
    )


def authorization() -> TransitionAuthorization:
    return TransitionAuthorization(
        "authorization-1",
        "proposal-1",
        "evaluation-1",
        "epoch-7",
        "release-owner",
        True,
        ("dual_run", "promote", "rollback"),
    )


def runtime(tmp_path) -> ArchitectureTransitionController:
    return ArchitectureTransitionController(
        tmp_path,
        governance_epoch="epoch-7",
        trusted_authorization_issuers=("release-owner",),
    )


def test_migration_is_separately_authorized_dual_run_and_reversible(tmp_path) -> None:
    controller = runtime(tmp_path)
    controller.submit(plan(), event_id="event-submit")
    controller.authorize("plan-1", authorization(), event_id="event-authorize")
    controller.start_dual_run("plan-1", event_id="event-start")
    controller.complete_dual_run(
        "plan-1", event_id="event-complete", passed=True, evidence_refs=("comparison-1", "comparison-2")
    )
    controller.rollback(
        "plan-1", event_id="event-rollback", reason="post-promotion regression", evidence_refs=("incident-1",)
    )
    assert controller.state("plan-1") is TransitionState.ROLLED_BACK
    assert controller.receipts[-1].evidence_refs == ("authorization-1", "incident-1")


def test_epoch_mismatch_and_protected_target_rejections_are_durable(tmp_path) -> None:
    controller = runtime(tmp_path)
    receipt = controller.submit(plan(epoch="epoch-6", targets=("evaluator",)), event_id="event-rejected")
    assert receipt.state is TransitionState.REJECTED
    restored = runtime(tmp_path)
    assert restored.state("plan-1") is TransitionState.REJECTED
    assert restored.receipts[-1] == receipt


def test_failed_rollout_and_duplicate_events_survive_restart(tmp_path) -> None:
    controller = runtime(tmp_path)
    submitted = controller.submit(plan(), event_id="event-submit")
    assert controller.submit(plan(), event_id="event-submit") == submitted
    controller.authorize("plan-1", authorization(), event_id="event-authorize")
    controller.start_dual_run("plan-1", event_id="event-start")
    failed = controller.complete_dual_run(
        "plan-1",
        event_id="event-fail",
        passed=False,
        evidence_refs=("comparison-failed-1", "comparison-failed-2"),
    )
    restored = runtime(tmp_path)
    assert restored.state("plan-1") is TransitionState.FAILED
    assert restored.receipts[-1] == failed


def test_out_of_order_event_does_not_duplicate_or_advance_a_migration(tmp_path) -> None:
    controller = runtime(tmp_path)
    controller.submit(plan(), event_id="event-submit")
    ignored = controller.start_dual_run("plan-1", event_id="event-early")
    assert ignored.recorded_action == "ignored_out_of_order"
    assert controller.state("plan-1") is TransitionState.SUBMITTED
    assert controller.start_dual_run("plan-1", event_id="event-early") == ignored
