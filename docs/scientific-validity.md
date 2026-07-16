# Scientific boundary

The control plane records that bytes were produced under a capability version, inputs, environment, policy, resources,
and run receipt. It does not infer that a proof is valid, a PDE converged, a posterior calibrated, a claim true, or a
model useful. Results remain `not_evaluated` until the owning domain and independent Harness evidence say otherwise.
Promotion policy consumes immutable verdict receipts but cannot manufacture them.

Operational monitoring preserves the same boundary. A threshold breach may quarantine or roll back a deployment for
safety or reliability, but it does not falsify a model or scientific claim. A healthy latency/error window does not
establish calibration, representativeness, applicability, causal validity, or scientific correctness. Domain-owned or
Harness-produced signals may be recorded as metrics only under an explicit policy; MLOps does not reinterpret them.

Registry integrity checking preserves the same boundary: it confirms the control plane's own bookkeeping is
self-consistent (a receipt log that replays to the live state, no dangling references, byte-exact artifacts), never
that a candidate is a good model. A clean integrity report is not promotion evidence and carries no calibration,
accuracy, or fitness claim.
