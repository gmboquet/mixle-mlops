# Semantic ownership migrations

The packaged control-surface manifest is canonical for this slice. Data ingestion and document/multimodal semantics
move to Mixle Data; model and synthesis semantics to Mixle Represent; training/evolution intent to AI Factory;
verification verdicts to Harness; knowledge semantics to Knowledge; domain physics and simulation to their owners; and
application-specific biodiversity code to Demos/plugins. MLOps retains hosted adapters, execution, storage, serving,
monitoring, and compatibility windows until each public replacement passes migration fixtures.

Registry files written before monitoring omit `quarantines`; they load as an empty quarantine map. Existing candidate,
alias, evidence, promotion, rollback, and receipt fields remain valid, and new monitoring metadata on rollback receipts
is optional. Once a candidate is quarantined, older software that ignores quarantine state must not operate the same
registry file; rollback is to revert the code and restore a pre-quarantine registry backup, not silently discard the
incident.
