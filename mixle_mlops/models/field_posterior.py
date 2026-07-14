"""``FieldPosteriorAdapter`` -- serve a fitted physics field posterior (IC-1 ``Posterior``) through the
platform's uniform ``ModelAdapter`` contract, mirroring ``MixleAdapter`` (models/mixle_model.py) but for a
spatial/volumetric physics posterior (e.g. a ``mixle_pde`` inversion result) instead of a mixle probabilistic
model.

The same posterior answers:

* ``predict`` -- a calibrated marginal slice over a caller-selected region (mean + credible interval),
* ``decide``  -- the Bayes-optimal action under a caller-supplied loss, evaluated on posterior draws of a
  region's derived quantity (default: region-mass, i.e. ``sum(field[region])``), reusing the same
  ``core.decision.bayes_action`` machinery ``MixleAdapter.decide`` already uses,
* ``score``   -- empirical coverage of held-out truth against this posterior's credible intervals.

A field posterior is not a chat model, so ``stream`` refuses (mirrors IC-7 ``DomainModelAdapter.stream``).

Note: this module type-checks against the frozen IC-1 ``Posterior`` protocol
(``mixle.reason.posterior_protocol``) but does not import it at runtime -- ``Posterior`` is a
``@runtime_checkable`` structural protocol, and any object exposing ``samples``/``mean``/``cov``/
``credible_interval``/``derived_quantity`` satisfies it whether or not that module happens to be installed.
Likewise, IC-2's ``mixle_pde.io.artifacts.load_posterior`` is imported lazily, only when ``posterior_ref=``
is actually used, so a caller who always passes ``posterior=`` directly has no hard dependency on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

import numpy as np

from ..core.adapters import CapabilityError, ChatCompletionChunk, ChatRequest, ModelAdapter
from ..core.decision import bayes_action

if TYPE_CHECKING:  # pragma: no cover -- IC-1 typing only; not required at import time
    from mixle.reason.posterior_protocol import Posterior


class _FixedDraws:
    """Adapts a precomputed draw array to the ``samples(n, rng)`` shape ``bayes_action`` expects.

    ``Posterior.derived_quantity`` (IC-1) already draws its own ``n`` samples; wrapping the resulting
    array here lets ``decide`` reuse ``core.decision.bayes_action`` unchanged instead of re-deriving a
    bespoke Bayes-action loop over a derived quantity.
    """

    def __init__(self, draws: np.ndarray) -> None:
        self._draws = np.asarray(draws)

    def samples(self, n: int, rng: Any) -> np.ndarray:
        if n == len(self._draws):
            return self._draws
        idx = rng.randint(0, len(self._draws), size=n)
        return self._draws[idx]


def _region_indices(record: Any, d: int) -> np.ndarray:
    """Resolve one ``records`` entry into an index array over the posterior's ``d`` field components.

    Accepts ``None``/``"all"`` (the whole field), a bare sequence of integer indices or a boolean mask,
    or a dict carrying ``indices`` (explicit indices), ``mask`` (boolean array), or ``slice``
    (``[start, stop, step]``).
    """
    if record is None or record == "all":
        return np.arange(d)
    if isinstance(record, dict):
        if "indices" in record:
            return np.asarray(record["indices"], dtype=int).reshape(-1)
        if "mask" in record:
            return np.nonzero(np.asarray(record["mask"], dtype=bool))[0]
        if "slice" in record:
            start, stop, step = (list(record["slice"]) + [None, None, None])[:3]
            return np.arange(d)[slice(start, stop, step)]
        raise ValueError(f"unrecognised region record: {record!r}")
    arr = np.asarray(record)
    if arr.dtype == bool:
        return np.nonzero(arr)[0]
    return arr.astype(int).reshape(-1)


class FieldPosteriorAdapter(ModelAdapter):
    """Serve one fitted IC-1 ``Posterior`` (typically a ``mixle_pde`` field posterior) as a hosted model.

    Args:
        name: the model id under which the gateway registers/serves it.
        posterior: a fitted object satisfying IC-1 ``Posterior`` (``samples``/``mean``/``cov``/
            ``credible_interval``/``derived_quantity``). Mutually exclusive with ``posterior_ref``/``registry``.
        posterior_ref: a content-hashed artifact handle, loaded via ``mixle_pde.io.artifacts.load_posterior``
            (IC-2) -- imported lazily so this module carries no hard dependency on ``mixle_pde``'s IO package.
        registry: a catalog exposing ``.load(name)`` (mirroring ``mixle.registry.Registry.load``) whose
            ``name`` entry is a ``field_posterior``-kind artifact (IC-2's registrable model kind).
    """

    kind = "field"

    def __init__(
        self,
        name: str,
        *,
        posterior: "Posterior | Any" = None,
        posterior_ref: str | None = None,
        registry: Any = None,
    ) -> None:
        self._name = name
        self._posterior = self._resolve(name, posterior, posterior_ref, registry)

    @staticmethod
    def _resolve(name: str, posterior: Any, posterior_ref: str | None, registry: Any) -> Any:
        if posterior is not None:
            return posterior
        if posterior_ref is not None:
            try:
                from mixle_pde.io.artifacts import load_posterior
            except ImportError as exc:
                raise ImportError(
                    "FieldPosteriorAdapter(posterior_ref=...) needs mixle_pde's artifact IO "
                    "(mixle_pde.io.artifacts.load_posterior, IC-2); install/land the mixle-pde 'io' package."
                ) from exc
            return load_posterior(posterior_ref)
        if registry is not None:
            return registry.load(name)
        raise ValueError("FieldPosteriorAdapter needs one of posterior=, posterior_ref=, or registry=")

    @property
    def name(self) -> str:
        return self._name

    # --- capability advertisement: a field posterior always answers all three (unlike MixleAdapter, which
    # gates on what the wrapped mixle model happens to support) ---
    def capabilities(self) -> set[str]:
        return {"predict", "decide", "score"}

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        """A field posterior is not a chat model (mirrors IC-7 ``DomainModelAdapter.stream``)."""
        raise CapabilityError(self._name, "stream")
        yield  # pragma: no cover -- makes this an async generator so the signature matches ModelAdapter

    async def predict(self, records: list[Any], **opts: Any) -> Any:
        """Calibrated marginal slice(s): mean + credible interval over each requested region.

        ``opts['level']`` sets the credible-interval coverage (default ``0.9``). A record that is a dict
        carrying ``section`` is forwarded to the posterior's own ``slice(...)`` when it exposes one;
        otherwise every record is resolved to field indices via :func:`_region_indices`.
        """
        level = float(opts.get("level", 0.9))
        mean = np.asarray(self._posterior.mean)
        lo, hi = self._posterior.credible_interval(level)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        out: list[dict[str, Any]] = []
        for record in records or [None]:
            if isinstance(record, dict) and "section" in record and hasattr(self._posterior, "slice"):
                out.append({"section": self._posterior.slice(**record["section"]), "level": level})
                continue
            idx = _region_indices(record, mean.shape[0])
            out.append(
                {
                    "indices": idx.tolist(),
                    "mean": mean[idx].tolist(),
                    "interval": (lo[idx].tolist(), hi[idx].tolist()),
                    "level": level,
                }
            )
        return {"records": out, "level": level}

    async def decide(self, records: list[Any], **opts: Any) -> Any:
        """Bayes-optimal action under ``opts['loss']`` over ``opts['actions']``, evaluated on posterior
        draws of a region's derived quantity (default: region-mass over the first record's region;
        override with ``opts['quantity_fn'](draws) -> (n,) array``). Mirrors ``MixleAdapter.decide``
        (mixle_model.py) but draws from the IC-1 ``derived_quantity`` pushforward rather than a plain
        predictive/parameter posterior.
        """
        loss = opts.get("loss")
        actions = opts.get("actions")
        if loss is None or actions is None:
            raise CapabilityError(self._name, "decide")
        n = int(opts.get("n", 2000))
        seed = int(opts.get("seed", 0))
        rng = np.random.default_rng(seed)
        mean = np.asarray(self._posterior.mean)
        region = records[0] if records else None
        idx = _region_indices(region, mean.shape[0])
        quantity_fn = opts.get("quantity_fn") or (lambda draws: draws[:, idx].sum(axis=1))
        dq = self._posterior.derived_quantity(quantity_fn, n, rng)
        draws = np.asarray(dq.samples)
        result = bayes_action(
            _FixedDraws(draws),
            loss,
            actions,
            n=len(draws),
            seed=seed,
            cvar_alpha=opts.get("cvar_alpha", 0.1),
        )
        result["prior_dominated"] = bool(dq.prior_dominated)
        return result

    async def score(self, records: list[Any], **opts: Any) -> Any:
        """Empirical coverage of held-out ``opts['truth']`` against this posterior's credible intervals."""
        truths = opts.get("truth")
        if truths is None:
            raise CapabilityError(self._name, "score")
        level = float(opts.get("level", 0.9))
        lo, hi = self._posterior.credible_interval(level)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        recs = records or [None] * len(truths)
        covered: list[bool] = []
        for record, truth in zip(recs, truths):
            idx = _region_indices(record, lo.shape[0])
            truth_arr = np.asarray(truth, dtype=float).reshape(-1)
            covered.append(bool(np.all((truth_arr >= lo[idx]) & (truth_arr <= hi[idx]))))
        coverage = float(np.mean(covered)) if covered else float("nan")
        return {"coverage": coverage, "nominal": level, "n": len(covered), "covered": covered}


def register_field_posterior(registry: Any, name: str, posterior: Any) -> FieldPosteriorAdapter:
    """Register a fitted field posterior with a ``ModelRegistry`` (mirrors ``register_demo_mixle_model``)."""
    adapter = FieldPosteriorAdapter(name, posterior=posterior)
    registry.register(adapter)
    return adapter
