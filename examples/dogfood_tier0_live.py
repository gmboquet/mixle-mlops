"""LIVE tier-0 dogfood — every seam real, end to end, in one command.

    python examples/dogfood_tier0_live.py [--agent-repo ~/codex/mixle-agent]

What actually happens (no fakes at any layer):

  1. mixle distills a tool-caller from a rigid teacher (selection = conformal solve, per-argument
     calibrated extractors) and saves the artifact under a scratch registry.
  2. A real mixle-mlops gateway boots on a free port (uvicorn subprocess) and serves it at
     /v1/toolcallers/{name} behind real auth (fresh /auth/signup key).
  3. The orchestrator picks, from FRESH traffic, one request the artifact provably answers locally
     and one it provably escalates (decided by the artifact itself before serving — no guessing).
  4. mixle-agent's REAL Tier0Router (TypeScript, real fetch) runs against the live gateway via
     `node --test tier0.live.test.ts`: the confident request must come back with the expected tool
     and args; the unsure one must escalate, and the posted "frontier" call must be accepted.
  5. The harvest is verified ON DISK in the registry — the trace that trains the next round.

Exit 0 = the dogfood loop is closed: agent -> tiny model -> tool call / escalate -> harvest.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np


def _teacher(request: str):
    import re

    m = re.search(r"weather (?:in|for) (\w+)", request)
    if m:
        return {"tool": "get_weather", "args": {"city": m.group(1)}}
    m = re.search(r"search for (.+)$", request)
    if m:
        return {"tool": "search", "args": {"query": m.group(1)}}
    return {"tool": None, "args": {}}


def _requests(n: int, seed: int = 0) -> list[str]:
    rng = np.random.RandomState(seed)
    cities = ["paris", "tokyo", "denver", "oslo"]
    out = []
    for _ in range(n):
        r = rng.rand()
        if r < 0.45:
            out.append(f"please tell me the weather in {cities[rng.randint(0, 4)]} today")
        elif r < 0.8:
            out.append(f"can you search for item {rng.randint(1000, 9999)}")
        else:
            out.append(f"thanks so much, note {rng.randint(0, 99)}")
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, body: dict | None = None, key: str | None = None) -> dict:
    req = urllib.request.Request(url, method=method)
    req.add_header("content-type", "application/json")
    if key:
        req.add_header("authorization", f"Bearer {key}")
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(req, data=data, timeout=15) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-repo", default=str(Path.home() / "codex" / "mixle-agent"))
    args = ap.parse_args()
    agent_core = Path(args.agent_repo).expanduser() / "packages" / "core"
    if not (agent_core / "test" / "tier0.live.test.ts").exists():
        print(f"mixle-agent not found at {args.agent_repo} (need packages/core/test/tier0.live.test.ts)")
        return 2

    from mixle.task import ToolSpec, distill_tool_caller

    with tempfile.TemporaryDirectory() as tmp:
        registry = Path(tmp) / "registry"

        print("[1/5] distilling the tool-caller (selection solve + per-arg extractors) ...")
        tools = [ToolSpec("get_weather", ["city"]), ToolSpec("search", ["query"])]
        tc = distill_tool_caller(
            _teacher,
            _requests(250),
            tools,
            seed=0,
            selector_kw={"ood": None, "epochs": 200},
            extractor_kw={"epochs": 30},
        )
        tc.save(str(registry / "toolcallers" / "assistant"))

        # the artifact itself decides which fresh requests it trusts — the test then must AGREE over HTTP
        confident = escalate = expected_tool = None
        for r in _requests(80, seed=7):
            local = tc.try_local(r)
            if local is not None and confident is None:
                confident, expected_tool = r, local["tool"]
            if local is None and escalate is None:
                escalate = r
            if confident and escalate:
                break
        if not (confident and escalate):
            print("fresh traffic produced no (confident, escalate) pair — enlarge the probe set")
            return 2
        print(f"      confident: {confident!r} -> {expected_tool}")
        print(f"      escalates: {escalate!r}")

        print("[2/5] booting the real gateway (uvicorn) ...")
        port = _free_port()
        env = dict(os.environ, MIXLE_DATA_DIR=tmp, MIXLE_REGISTRY_ROOT=str(registry))
        gw = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "mixle_mlops.gateway.app:app", "--port", str(port), "--log-level", "warning"],
            env=env,
            cwd=Path(__file__).resolve().parents[1],
        )
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(120):
                try:
                    _http("GET", f"{base}/health")
                    break
                except Exception:  # noqa: BLE001 - still booting
                    time.sleep(0.25)
            else:
                print("gateway never became healthy")
                return 2

            key = _http("POST", f"{base}/auth/signup", {"email": "dogfood@local", "password": "pw12345"})["api_key"]
            served = _http("GET", f"{base}/v1/toolcallers", key=key)
            assert served == {"toolcallers": ["assistant"]}, served
            print(f"[3/5] gateway healthy on :{port}, artifact listed, auth live")

            print("[4/5] running mixle-agent's REAL Tier0Router against it (node --test) ...")
            node_env = dict(
                os.environ,
                MIXLE_MLOPS_URL=base,
                MIXLE_MLOPS_KEY=key,
                MIXLE_TOOLCALLER="assistant",
                MIXLE_CONFIDENT_INPUT=confident,
                MIXLE_EXPECTED_TOOL=expected_tool,
                MIXLE_ESCALATE_INPUT=escalate,
            )
            node = subprocess.run(
                ["node", "--import", "tsx", "--test", "test/tier0.live.test.ts"],
                cwd=agent_core,
                env=node_env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            sys.stdout.write(node.stdout[-2000:])
            if node.returncode != 0:
                sys.stderr.write(node.stderr[-2000:])
                print("node --test FAILED")
                return 1

            harvested = registry / "toolcallers" / "assistant" / "harvested.jsonl"
            trace = json.loads(harvested.read_text().splitlines()[-1])
            assert trace["input"] == escalate and trace["call"]["tool"] == "search", trace
            print("[5/5] harvest verified on disk — the escalated trace is queued for the next round")
            print("\nDOGFOOD LOOP CLOSED: agent -> tiny model -> {tool call | escalate} -> harvest. All real.")
            return 0
        finally:
            gw.terminate()
            gw.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
