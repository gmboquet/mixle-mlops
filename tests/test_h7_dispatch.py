"""H7 -- dispatch / control policy.

H8's real digital twin has not landed (see mixle_mlops/dispatch_policy.py's module docstring), so this
exercises train_dispatch_policy/DispatchPolicy against a small synthetic, fixed-seed pipeline stand-in
that implements the duck-typed twin surface (`reset(seed)` / `step(action)`) train_dispatch_policy
expects. The twin models three shovels of equal service capacity feeding one plant, with a random batch
of trucks arriving each step that must be routed (as a whole) to one shovel; a policy that balances
routing across shovels keeps total queue low, while always routing to the same ("nearest") shovel
overloads it while the other two idle -- exactly the gap a learned dispatch policy should close.
"""

from __future__ import annotations

import random

from mixle_mlops.dispatch_policy import (
    DispatchPolicy,
    greedy_nearest_shovel,
    train_dispatch_policy,
)


class FixedSeedPipelineTwin:
    """A small, fully deterministic (given a seed) stand-in for the H8 digital twin.

    State: {"queues": {shovel_id: float}, "plant_feed": {plant_id: float}, "grades": {shovel_id: float}}.
    Action: {"shovel": shovel_id, "plant": plant_id} -- routes this step's whole arrival batch.
    """

    SHOVEL_SERVICE_RATE = 1.0
    QUEUE_PENALTY = 0.3

    def __init__(self, *, n_shovels: int = 3, episode_length: int = 30, max_arrivals: int = 3):
        self.shovels = [f"s{i}" for i in range(n_shovels)]
        self.plants = ["p0"]
        self.episode_length = episode_length
        self.max_arrivals = max_arrivals
        self._grades = {s: 1.0 + 0.1 * i for i, s in enumerate(self.shovels)}
        self._rng: random.Random | None = None
        self._queues: dict[str, float] = {}
        self._feed: dict[str, float] = {}
        self._step = 0

    def reset(self, seed: int | None = None) -> dict:
        self._rng = random.Random(seed)
        self._queues = {s: 0.0 for s in self.shovels}
        self._feed = {p: 0.0 for p in self.plants}
        self._step = 0
        return self._state()

    def step(self, action: dict) -> tuple[dict, float, bool]:
        assert self._rng is not None, "call reset() before step()"
        shovel = action["shovel"]
        plant = action["plant"]

        arrivals = self._rng.randint(0, self.max_arrivals)
        self._queues[shovel] = self._queues.get(shovel, 0.0) + arrivals
        if arrivals:
            self._feed[plant] = self._feed.get(plant, 0.0) + 1.0

        throughput = 0.0
        for s in self.shovels:
            served = min(self._queues[s], self.SHOVEL_SERVICE_RATE)
            self._queues[s] -= served
            throughput += served

        total_queue = sum(self._queues.values())
        reward = throughput - self.QUEUE_PENALTY * total_queue

        self._step += 1
        done = self._step >= self.episode_length
        return self._state(), reward, done

    def _state(self) -> dict:
        return {
            "queues": dict(self._queues),
            "plant_feed": dict(self._feed),
            "grades": dict(self._grades),
        }


def _mean_queue_time(twin: FixedSeedPipelineTwin, act_fn, seeds) -> float:
    """Time-averaged total queue length over each held-out episode, averaged across ``seeds``."""
    episode_means = []
    for seed in seeds:
        state = twin.reset(seed=seed)
        done = False
        queue_area = 0.0
        steps = 0
        while not done:
            action = act_fn(state)
            state, _reward, done = twin.step(action)
            queue_area += sum(state["queues"].values())
            steps += 1
        episode_means.append(queue_area / steps)
    return sum(episode_means) / len(episode_means)


def test_train_dispatch_policy_returns_dispatch_policy():
    twin = FixedSeedPipelineTwin()
    policy = train_dispatch_policy(twin, episodes=20, seed=1)
    assert isinstance(policy, DispatchPolicy)
    assert policy.trained_episodes == 20
    state = twin.reset(seed=0)
    action = policy.act(state)
    assert set(action) == {"shovel", "plant"}
    assert action["shovel"] in twin.shovels
    assert action["plant"] in twin.plants


def test_act_falls_back_to_greedy_nearest_shovel_for_unseen_state():
    twin = FixedSeedPipelineTwin()
    state0 = twin.reset(seed=0)
    policy = DispatchPolicy(q_table={}, actions=[{"shovel": s, "plant": "p0"} for s in twin.shovels])
    assert policy.act(state0) == greedy_nearest_shovel(state0, policy.actions)


def test_register_returns_ic10_shaped_catalog_entry():
    twin = FixedSeedPipelineTwin()
    policy = train_dispatch_policy(twin, episodes=10, seed=2)
    entry = policy.register()
    assert set(entry) == {"id", "schema", "owner", "cost", "reliability", "verifier"}
    assert entry["owner"] == "dispatch"
    assert 0.0 <= entry["reliability"] <= 1.0
    assert isinstance(entry["schema"], dict)


def test_trained_policy_beats_greedy_nearest_shovel_baseline_on_mean_queue_time():
    twin = FixedSeedPipelineTwin(n_shovels=3, episode_length=30, max_arrivals=3)

    policy = train_dispatch_policy(twin, episodes=400, seed=7)

    baseline_actions = [{"shovel": s, "plant": "p0"} for s in twin.shovels]
    held_out_seeds = range(90_000, 90_000 + 40)  # disjoint from training's seed*1_000_003+episode stream

    trained_mean = _mean_queue_time(twin, policy.act, held_out_seeds)
    baseline_mean = _mean_queue_time(twin, lambda s: greedy_nearest_shovel(s, baseline_actions), held_out_seeds)

    assert trained_mean < baseline_mean
    # not a marginal win -- the trained policy should hold mean queue time to well under baseline
    assert trained_mean <= 0.6 * baseline_mean
