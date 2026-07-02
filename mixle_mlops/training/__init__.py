"""User-facing training: fine-tune a model through the gateway and serve the result.

``POST /v1/fine_tunes`` turns a labeled dataset into a hosted model. The ``structured`` backend distils a tiny,
interpretable, torch-free structured probabilistic classifier locally (``mixle.task.distill_structured``) and
registers it into the live model registry, so it appears in ``/v1/models`` and answers ``/v1/chat`` and
``/v1/mixle/{predict,score}`` immediately -- the whole loop, no GPU. The ``llm``/``mixle`` backends produce a
vast.ai training *plan* (offline, free) over :mod:`mixle_mlops.compute`; actually renting the GPU is the operator's
gated, keyed step.
"""
from .models import FineTuneJob
from .service import plan_gpu_job, run_structured_finetune

__all__ = ["FineTuneJob", "plan_gpu_job", "run_structured_finetune"]
