"""mixle-mlops: an all-in-one AI platform.

Host mixle's probabilistic models *and* open LLMs (Llama, DeepSeek, ...) behind one OpenAI-compatible gateway,
let mixle compose them, and ship the full product surface — accounts, API keys, multimodal I/O, an MCP server,
a chat UI + landing page, and a principled (mixle-powered) RLHF feedback loop. See ARCHITECTURE.md.

    from mixle_mlops.gateway import create_app
    app = create_app()              # a FastAPI app; run with `mixle-serve`
"""

from .context_handoff import ContextEvent, ContextRun, ContextRunState, ContextRunStore, bundle_digest
from .architecture import (
    ArchitectureTransitionController,
    ArchitectureTransitionError,
    ConsumerMigration,
    DualRunPolicy,
    MigrationPlan,
    ProviderContract,
    RollbackPlan,
    TransitionAuthorization,
    TransitionReceipt,
    TransitionState,
)
from .control import (
    DeploymentRegistry,
    DurableLocalRunner,
    EvidenceReceipt,
    InvocationSpec,
    JobSpec,
    LocalArtifactStore,
    ModelCandidate,
    PromotionPolicy,
)

__version__ = "0.8.0.dev0"

__all__ = [
    "ContextEvent",
    "ArchitectureTransitionController",
    "ArchitectureTransitionError",
    "ConsumerMigration",
    "ContextRun",
    "ContextRunState",
    "ContextRunStore",
    "DeploymentRegistry",
    "DurableLocalRunner",
    "DualRunPolicy",
    "EvidenceReceipt",
    "InvocationSpec",
    "JobSpec",
    "LocalArtifactStore",
    "ModelCandidate",
    "MigrationPlan",
    "PromotionPolicy",
    "ProviderContract",
    "RollbackPlan",
    "TransitionAuthorization",
    "TransitionReceipt",
    "TransitionState",
    "bundle_digest",
]
