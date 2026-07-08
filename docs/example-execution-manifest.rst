Example Execution Manifest
==========================

``mixle-mlops`` examples exercise serving, multimodal, GPU, structured-fusion,
and local platform workflows. They are useful release evidence only when their
runtime assumptions are explicit.

Current Inventory
-----------------

The package currently ships 11 Python example scripts plus one companion
README in the neural-experts example directory:

.. list-table::
   :header-rows: 1

   * - Path
     - Release status required
   * - ``examples/platform_tour.py``
     - Execute from a clean install or record backend/config blockers.
   * - ``examples/dogfood_tier0_live.py``
     - Manual/live-provider run unless credentials and endpoint are available.
   * - ``examples/train_tiny_parser_lm.py``
     - Execute or record model-training runtime limits.
   * - ``examples/_embed_cifar100_clip.py``
     - Helper script; execute only when CLIP/CIFAR assets are provisioned.
   * - ``examples/adapt_vlm_structured.py``
     - Manual or blocked on VLM dependencies.
   * - ``examples/distill_vlm_to_structured.py``
     - Manual or blocked on VLM/student dependencies.
   * - ``examples/structured_fusion_cifar.py``
     - Manual or blocked on dataset/model dependencies.
   * - ``examples/structured_fusion_vlm.py``
     - Manual or blocked on VLM dependencies.
   * - ``examples/gpu_smoketest/gpu_check.py``
     - Execute on GPU runners; blocked on CPU-only runners.
   * - ``examples/torch_engine_gpu/train.py``
     - Execute on GPU runners; blocked on CPU-only runners.
   * - ``examples/mixle_neural_experts/train.py``
     - Execute or record neural-training runtime limits.
   * - ``examples/mixle_neural_experts/``
     - Keep the companion README synchronized with the training script and
       release notes.

Release Execution Status
------------------------

The current documentation state does not prove these examples execute
end-to-end. Before a public release, each example must be recorded as
``passed``, ``failed``, ``timed_out``, ``blocked``, or ``skipped`` with the
reason.

Gateway examples must also record:

* server start command;
* storage/cache backend configuration;
* provider/model configuration;
* whether credentials were example-only, local, or real;
* route or CLI command exercised; and
* output artifacts or logs produced.

GPU examples must record hardware, driver/runtime, Torch version, and whether
the run was a smoke check or a training-quality run.

Minimum Release Run
-------------------

At minimum, run or classify:

* ``platform_tour.py`` for the local platform path;
* ``train_tiny_parser_lm.py`` for local training;
* ``gpu_smoketest/gpu_check.py`` on GPU CI or blocked on CPU-only CI;
* one structured-fusion example if VLM dependencies are in the release
  environment; and
* ``mixle_neural_experts/train.py`` or an explicitly bounded smoke variant.

Do not count live-provider examples as passed unless the actual provider
request path was exercised and recorded.
