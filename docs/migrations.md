# Semantic ownership migrations

The packaged control-surface manifest is canonical for this slice. Data ingestion and document/multimodal semantics
move to Mixle Data; model and synthesis semantics to Mixle Represent; training/evolution intent to AI Factory;
verification verdicts to Harness; knowledge semantics to Knowledge; domain physics and simulation to their owners; and
application-specific biodiversity code to Demos/plugins. MLOps retains hosted adapters, execution, storage, serving,
monitoring, and compatibility windows until each public replacement passes migration fixtures.

The integrity checker reads only the existing `deployments.json`/artifact-store schema; it introduces no new
persisted fields and requires no migration. It is safe to run against a registry written by any prior control-kernel
release, including one predating this checker.
