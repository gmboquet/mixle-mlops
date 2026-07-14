"""H7 -- dispatch / control policy for the mine -> plant -> distribution pipeline.

Trains a tabular Q-learning dispatch policy against a digital-twin simulation (H8) of the pipeline and
serves the result as an IC-10 catalog entry, so the mlops router (M3) can call ``DispatchPolicy.act``
uniformly alongside the platform's other tools/models. The policy answers one question, fast: given the
twin's current state (queue lengths / plant-feed levels / grades), which truck->shovel and plant-feed
assignment should we make right now? This is near-real-time re-dispatch, not a sub-second control loop
and not online learning in production (training happens once, offline, against the twin).

Environment note (read before touching the twin plumbing)
-----------------------------------------------------------
H8's public surface (per ``notes/exec/workstream-H.md``) is a *period-batch* interface --
``PipelineTwin.run(n_periods, *, scenario=None) -> dict`` -- not a step-wise RL loop, and at the time
this module was written H8 had not landed (no ``mixle.pipeline_twin`` module exists yet; H1's
``mixle.relations`` flow primitives that a real twin would use were also still on an unmerged branch).
Tabular Q-learning needs one ``(state, action, reward, next_state)`` transition at a time, so
``train_dispatch_policy`` is written against a minimal step-wise duck type instead of importing a
concrete twin class -- which is exactly what the frozen ``twin: Any`` signature already commits to:

    state = twin.reset(seed=...)          # dict: at least "queues" and "plant_feed" (both {id: float}),
                                           # optionally "grades" ({id: float})
    next_state, reward, done = twin.step(action)   # action: {"shovel": id, "plant": id}

Any object exposing that surface trains here, including a future ``PipelineTwin`` once it grows a
``step`` method that wraps its own period re-solve, or (as in this package's tests) a small synthetic
stand-in used as the "fixed-seed twin" the Definition of Done calls for. Nothing here special-cases a
twin class name or reaches into ``mixle.relations``/``mixle.pipeline_twin`` directly, so it degrades
gracefully to whatever twin -- toy or real -- is handed in.

``mixle.inference.decision`` (Bayes-optimal actions under a fitted *posterior*) and
``mixle.inference.planning`` (EM estimation scheduling) do not, in the 0.8.0 tree, expose a tabular
Q-learning or max-ent IRL surface -- the "0.7.0 tabular Q-learning / max-ent IRL surface" the work order
describes is not present in either module. Forcing this policy through ``bayes_action`` would mean
manufacturing a fake posterior over Q-values just to satisfy an import; that is worse than being direct
about the gap. The Q-learning update below is implemented directly against the twin's simulated reward,
which is what the algorithm actually calls for (DR-ALG-equivalent: tabular Q-learning / SARSA-style
bootstrapped update over a discretized state space).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DispatchPolicy", "train_dispatch_policy", "greedy_nearest_shovel"]


Action = dict[str, Any]
State = dict[str, Any]


def _round_bucket(value: Any, *, digits: int = 0, cap: float | None = None) -> Any:
    """Discretize one state-field value into a hashable, generalizing tabular-Q bucket."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if cap is not None:
        v = min(v, cap)
    return round(v, digits)


def _state_key(state: State, *, queue_cap: float = 12.0) -> tuple:
    """Map a (possibly continuous) state dict to a hashable tabular-Q key.

    Queue lengths and plant-feed levels are rounded to whole units and capped so nearby/overflowing
    states share a Q-row (a state space that stays small enough for 100s of training episodes to
    cover); grades (if present) are rounded to one decimal -- they matter for the *value* of an
    action, not for identifying "the same" congestion state.
    """
    queues = tuple(sorted((k, _round_bucket(v, cap=queue_cap)) for k, v in state.get("queues", {}).items()))
    feed = tuple(sorted((k, _round_bucket(v, cap=queue_cap)) for k, v in state.get("plant_feed", {}).items()))
    grades = tuple(sorted((k, _round_bucket(v, digits=1)) for k, v in state.get("grades", {}).items()))
    return (queues, feed, grades)


def _action_key(action: Action) -> tuple:
    return tuple(sorted(action.items()))


def _action_from_key(key: tuple, actions: Sequence[Action]) -> Action:
    for a in actions:
        if _action_key(a) == key:
            return a
    return actions[0]


def _derive_actions(state: State) -> list[Action]:
    """The action set: every (shovel, plant) pair the twin's own state names -- truck->shovel plus
    plant-feed assignment, per the algorithm spec. Falls back to a single no-op-shaped action if a
    twin reports no shovels/plants (degenerate but keeps `act` total)."""
    shovels = list(state.get("queues", {})) or ["_default_shovel"]
    plants = list(state.get("plant_feed", {})) or ["_default_plant"]
    return [{"shovel": s, "plant": p} for s in shovels for p in plants]


def greedy_nearest_shovel(state: State, actions: Sequence[Action]) -> Action:
    """The naive dispatch baseline: always the (spatially) nearest shovel/plant -- the first entry in
    ``actions``' natural order -- regardless of current queue state. This is exactly the heuristic a
    state-aware dispatch policy should beat, and it doubles as ``DispatchPolicy``'s fallback for any
    state it never saw during training."""
    if not actions:
        raise ValueError("greedy_nearest_shovel requires at least one candidate action")
    return actions[0]


@dataclass
class DispatchPolicy:
    """A tabular-Q dispatch policy over truck->shovel / plant-feed assignments.

    ``q_table`` maps a discretized state key to ``{action_key: value}``; ``act`` is an O(1) dict
    lookup (the near-real-time re-dispatch call the algorithm calls for -- no re-solve per call).
    Unseen states fall back to :func:`greedy_nearest_shovel` rather than raising, so ``act`` is total.
    """

    q_table: dict[tuple, dict[tuple, float]]
    actions: list[Action] = field(default_factory=list)
    seed: int = 0
    trained_episodes: int = 0

    def act(self, state: State) -> Action:
        row = self.q_table.get(_state_key(state))
        if not row:
            return greedy_nearest_shovel(state, self.actions)
        best_key = max(row, key=row.get)
        return _action_from_key(best_key, self.actions)

    def register(self) -> dict:
        """The IC-10 catalog entry (``{id, schema, owner, cost, reliability, verifier}``) so M3 routing
        sees this policy the same way it sees every other tool/model. Returned as a plain dict (not a
        ``mixle.task.catalog.CatalogEntry`` instance) since that module is itself still an unfilled
        contract elsewhere in the tree; the shape matches it field-for-field so a caller can do
        ``CatalogEntry(**policy.register())`` once it exists.
        """
        visited = sum(1 for row in self.q_table.values() if any(v != 0.0 for v in row.values()))
        reliability = 0.5 if not self.q_table else min(0.95, 0.5 + 0.45 * (visited / len(self.q_table)))
        return {
            "id": "dispatch-policy-v1",
            "schema": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "object",
                        "description": "queues / plant_feed / grades snapshot from the pipeline twin",
                        "properties": {
                            "queues": {"type": "object"},
                            "plant_feed": {"type": "object"},
                            "grades": {"type": "object"},
                        },
                        "required": ["queues", "plant_feed"],
                    }
                },
                "required": ["state"],
                "returns": {
                    "type": "object",
                    "properties": {"shovel": {"type": "string"}, "plant": {"type": "string"}},
                },
            },
            "owner": "dispatch",
            "cost": 0.0,
            "reliability": reliability,
            "verifier": None,
        }


def train_dispatch_policy(twin: Any, *, episodes: int = 500, seed: int = 0) -> DispatchPolicy:
    """Train a tabular Q-learning dispatch policy against ``twin``'s simulated reward.

    ``twin`` must expose ``reset(seed=...) -> state`` and ``step(action) -> (next_state, reward,
    done)`` -- see the module docstring for the exact duck type expected of the H8 twin. Each episode
    resets the twin with a seed derived from ``seed`` and the episode index (reproducible given a
    fixed-seed twin), rolls out an epsilon-greedy policy over the twin's own reward signal
    (throughput minus queue penalty, computed by the twin), and applies a standard bootstrapped
    Q-update after every transition.
    """
    rng = random.Random(seed)
    probe_state = twin.reset(seed=seed)
    actions = _derive_actions(probe_state)

    q_table: dict[tuple, dict[tuple, float]] = {}
    alpha = 0.3  # learning rate
    gamma = 0.9  # discount
    max_steps_per_episode = 500  # guard against a twin whose `done` never fires
    epsilon_floor = 0.05

    def q_row(state: State) -> dict[tuple, float]:
        return q_table.setdefault(_state_key(state), {_action_key(a): 0.0 for a in actions})

    for episode in range(episodes):
        state = twin.reset(seed=seed * 1_000_003 + episode)
        epsilon = max(epsilon_floor, 1.0 - episode / max(1, episodes * 0.8))
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            row = q_row(state)
            if rng.random() < epsilon:
                action = actions[rng.randrange(len(actions))]
            else:
                action = _action_from_key(max(row, key=row.get), actions)

            next_state, reward, done = twin.step(action)
            next_row = q_row(next_state)
            best_next = max(next_row.values()) if next_row else 0.0

            a_key = _action_key(action)
            row[a_key] += alpha * (float(reward) + gamma * best_next - row[a_key])

            state = next_state
            steps += 1

    return DispatchPolicy(q_table=q_table, actions=actions, seed=seed, trained_episodes=episodes)
