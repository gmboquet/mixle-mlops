"""Minimal Vast.ai REST client.

Wraps the documented API (``https://console.vast.ai/api/v0``, Bearer auth): search offers, create an
instance from an offer, read instance status (ssh host/port), and destroy it. Just the calls the
training launcher needs — no SDK dependency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

API = "https://console.vast.ai/api/v0"


class VastError(Exception):
    pass


@dataclass
class Offer:
    id: int
    gpu_name: str
    num_gpus: int
    price: float  # $/hr (dph_total)
    reliability: float = 0.0  # vast reliability2 score in [0,1] — flaky cheap hosts score low
    raw: dict = field(default_factory=dict)


class VastClient:
    def __init__(self, api_key: str, base: str = API, timeout: float = 30.0) -> None:
        if not api_key:
            raise VastError("a vast.ai API key is required (set MIXLE_VAST_API_KEY)")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    def _request(self, method: str, path: str, **kw) -> dict:
        try:
            r = httpx.request(method, f"{self.base}{path}", headers=self._headers, timeout=self.timeout, **kw)
        except httpx.HTTPError as e:
            raise VastError(f"vast.ai request failed ({method} {path}): {e}") from e
        if r.status_code >= 400:
            raise VastError(f"vast.ai {method} {path} -> {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            return {}

    @staticmethod
    def build_query(
        *, gpu_name: str | None, num_gpus: int, max_price: float | None, limit: int, min_reliability: float = 0.95
    ) -> dict:
        """The offer-search filter body (exposed for testing/dry-run).

        Filters to exactly ``num_gpus`` GPUs (so a 1-GPU job doesn't rent a pricier multi-GPU box) and to
        reliable hosts (``reliability2 >= min_reliability``) — the cheapest offers are often flaky hosts
        that never leave "loading"."""
        q: dict = {
            "rentable": {"eq": True},
            "num_gpus": {"eq": num_gpus},
            "type": "ondemand",
            "reliability2": {"gte": min_reliability},
            "order": [["dph_total", "asc"]],
            "limit": limit,
        }
        if gpu_name:
            q["gpu_name"] = {"in": [gpu_name.replace("_", " ")]}
        if max_price is not None:
            q["dph_total"] = {"lte": max_price}
        return q

    def search_offers(
        self, *, gpu_name: str | None = None, num_gpus: int = 1, max_price: float | None = None, limit: int = 20,
        min_reliability: float = 0.95,
    ) -> list[Offer]:
        data = self._request("POST", "/bundles/", json=self.build_query(
            gpu_name=gpu_name, num_gpus=num_gpus, max_price=max_price, limit=limit, min_reliability=min_reliability))
        offers: list[Offer] = []
        for o in data.get("offers", []):
            offers.append(
                Offer(
                    id=int(o.get("id", 0)),
                    gpu_name=str(o.get("gpu_name", "")),
                    num_gpus=int(o.get("num_gpus", 0)),
                    price=float(o.get("dph_total") or 0.0),
                    reliability=float(o.get("reliability2") or 0.0),
                    raw=o,
                )
            )
        return offers

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk: int,
        onstart: str | None = None,
        runtype: str = "ssh_direct",
        env: dict | None = None,
        label: str | None = None,
    ) -> int:
        body: dict = {"client_id": "me", "image": image, "disk": disk, "runtype": runtype}
        if onstart:
            body["onstart"] = onstart
        if env:
            body["env"] = env
        if label:
            body["label"] = label
        data = self._request("PUT", f"/asks/{offer_id}/", json=body)
        cid = data.get("new_contract")
        if not cid:
            raise VastError(f"vast.ai did not return a new instance id: {data}")
        return int(cid)

    def instance(self, instance_id: int) -> dict:
        data = self._request("GET", f"/instances/{instance_id}/")
        inst = data.get("instances", data)
        if isinstance(inst, list):
            for i in inst:
                if str(i.get("id")) == str(instance_id):
                    return i
            return {}
        return inst if isinstance(inst, dict) else {}

    def destroy(self, instance_id: int) -> None:
        self._request("DELETE", f"/instances/{instance_id}/")

    def is_gone(self, instance_id: int) -> bool:
        """Authoritative per-instance check: True once the instance no longer exists (or has terminated).

        The /instances/ LIST endpoint can lag and report an instance gone while it is still billing, so
        destruction must be confirmed against the per-instance GET, not the list."""
        try:
            inst = self.instance(instance_id)
        except VastError:
            return False  # couldn't check — treat as still-present so the caller keeps trying
        if not inst:
            return True
        status = str(inst.get("actual_status") or inst.get("cur_state") or "").lower()
        return status in ("exited", "destroyed")

    def destroy_confirmed(self, instance_id: int, on_log=None, timeout: float = 150.0, poll: float = 5.0) -> bool:
        """DELETE the instance and CONFIRM it is actually gone. A single DELETE can return 2xx without
        terminating the box promptly, so re-issue and poll the per-instance GET until it disappears.
        Returns True if confirmed gone within ``timeout``; False otherwise (caller must alert the user)."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.destroy(instance_id)
            except VastError as e:
                if on_log:
                    on_log(f"  destroy call error for {instance_id}: {e}")
            if self.is_gone(instance_id):
                return True
            if time.monotonic() >= deadline:
                return False
            if on_log:
                on_log(f"  instance {instance_id} not yet terminated; re-issuing destroy …")
            time.sleep(poll)
