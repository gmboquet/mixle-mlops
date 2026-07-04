"""``MixleAdapter`` -- serve a *fitted mixle probabilistic model* through the platform's uniform
``ModelAdapter`` contract, exposing the distribution/decision surface that makes this platform more than
an LLM proxy.

The same fitted model answers:

* ``predict``  -- a calibrated predictive distribution per record (mean/quantiles/interval/ensemble/CDF),
* ``score``    -- per-record log-density (the model's own likelihood; via a mixle ``Service`` when given),
* ``latent``   -- the latent posterior ``q(z|x)`` (mixture responsibilities / HMM Viterbi-or-smoothing),
* ``decide``   -- the Bayes-optimal action under a caller-supplied loss + a tail-risk profile,
* ``chat``     -- a concise text/JSON summary of ``predict`` so the model is reachable via ``/v1/chat`` too.

``capabilities()`` advertises only what the wrapped model actually supports, gated on
``mixle.capability``, so the gateway never routes a query a model cannot answer.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

import numpy as np

from mixle import capability as cap

from ..core.adapters import (
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatRequest,
    ChoiceDelta,
    CapabilityError,
    ModelAdapter,
    ModelInfo,
)
from ..core.decision import bayes_action
from ..core.predictive import predictive_batch


class MixleAdapter(ModelAdapter):
    """Wrap a fitted mixle ``Distribution`` (passed directly, or loaded from a production
    ``Registry``/``Service``) as a hosted model.

    Args:
        name: the model id under which the gateway registers/serves it.
        model: a fitted mixle distribution. Mutually exclusive with ``service``/``registry``.
        service: a ``mixle.inference.production.Service`` -- its ``.model`` is served and its
            ``.score`` is used for the score route (carries activity logging).
        registry: a ``mixle.inference.production.Registry`` to load ``name``'s ``alias`` model from.
        alias: which registry alias to serve (default ``"production"``).
        fit_data: optional training data retained so ``decide`` can build a parameter posterior
            ``q(theta | data)``; without it, ``decide`` uses the plug-in predictive posterior.
    """

    kind = "mixle"

    def __init__(
        self,
        name: str,
        model: Any = None,
        *,
        service: Any = None,
        registry: Any = None,
        alias: str = "production",
        fit_data: Sequence[Any] | None = None,
    ) -> None:
        self._name = name
        self._service = service
        if model is not None:
            self._model = model
        elif service is not None:
            self._model = service.model
        elif registry is not None:
            loaded, _header = registry.current(name, alias)
            self._model = loaded
        else:
            raise ValueError("MixleAdapter needs one of model=, service=, or registry=")
        self._fit_data = list(fit_data) if fit_data is not None else None

    @property
    def name(self) -> str:
        return self._name

    def info(self) -> ModelInfo:
        """The served manifest, enriched with the model's estimation certificate + calibration (I2).

        A hosted mixle model advertises not just its id and capabilities but HOW it was estimated: the
        certificate (guarantee ladder + why-not-ADAM audit) and, when the fit reserved a holdout, whether
        its uncertainty was judged calibrated. The certificate is read from ``model.certificate`` if the
        fit attached one, else computed on demand via :func:`mixle.inference.certify`. Absence is honest
        (a model that cannot be certified simply reports ``certificate: null``), never a crash."""
        base = super().info()
        base.certificate = self._certificate_manifest()
        base.calibration = self._calibration_manifest()
        return base

    def _certificate_manifest(self) -> dict[str, Any] | None:
        try:
            cert = getattr(self._model, "certificate", None)
            if cert is None:
                from mixle.inference import certify

                cert = certify(self._model)
            guarantee = cert.guarantee
            return {
                "guarantee": getattr(guarantee, "name", str(guarantee)),
                "guarantee_level": int(guarantee),
                "why_not_adam": cert.why_not_adam() if hasattr(cert, "why_not_adam") else None,
                "n_blocks": len(cert.blocks) if hasattr(cert, "blocks") else None,
                "n_gradient_blocks": len(cert.gradient_blocks) if hasattr(cert, "gradient_blocks") else None,
            }
        except Exception:  # noqa: BLE001 - a manifest must never fail to serve; absence is honest
            return None

    def _calibration_manifest(self) -> dict[str, Any] | None:
        calib = getattr(self._model, "calibration", None)
        if calib is None:
            return None
        try:
            return {
                "is_calibrated": bool(calib.is_calibrated()),
                "n_holdout": getattr(calib, "n", None),
            }
        except Exception:  # noqa: BLE001
            return None

    # --- capability advertisement (gated on what the wrapped model supports) ---
    def capabilities(self) -> set[str]:
        caps: set[str] = {"chat"}  # chat is always available (summarizes predict)
        m = self._model
        if cap.supports(m, cap.HasCDF) or callable(getattr(m, "sampler", None)):
            caps.add("predict")
        if callable(getattr(m, "log_density", None)) or callable(getattr(m, "seq_log_density", None)):
            caps.add("score")
        if cap.supports(m, cap.LatentStructured) or callable(getattr(m, "viterbi", None)):
            caps.add("latent")
        if callable(getattr(m, "sampler", None)):  # decide needs a posterior we can sample
            caps.add("decide")
        return caps

    # --- OpenAI-compatible chat: render a concise summary of predict() for the last user turn ---
    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        text = self._chat_summary(req)
        cid = f"chatcmpl-{self._name}"
        yield ChatCompletionChunk(
            id=cid,
            model=req.model,
            choices=[ChatChunkChoice(delta=ChoiceDelta(role="assistant"))],
        )
        yield ChatCompletionChunk(
            id=cid,
            model=req.model,
            choices=[ChatChunkChoice(delta=ChoiceDelta(content=text))],
        )
        yield ChatCompletionChunk(
            id=cid,
            model=req.model,
            choices=[ChatChunkChoice(delta=ChoiceDelta(), finish_reason="stop")],
        )

    def _chat_summary(self, req: ChatRequest) -> str:
        """Summarize predict() of the last user message (parsed as JSON record(s), else unconditional)."""
        last = next((m for m in reversed(req.messages) if m.role == "user"), None)
        raw = last.text().strip() if last else ""
        records = self._parse_records(raw)
        try:
            preds = predictive_batch(self._model, records)
        except CapabilityError as exc:
            return f"[{self._name}] cannot produce a predictive distribution: {exc}"
        lines = [
            f"mixle model '{self._name}' predictive distribution ({preds.path}, density={preds.density_semantics}):"
        ]
        for i, r in enumerate(preds.records):
            mean = "n/a" if r.mean is None else f"{r.mean:.4g}"
            q = r.quantiles
            med = q.get(0.5)
            med_s = f"{med:.4g}" if med is not None else "n/a"
            lo, hi = r.interval
            lines.append(
                f"  record[{i}]: mean={mean}, median={med_s}, "
                f"{int(preds.interval_level * 100)}% interval=[{lo:.4g}, {hi:.4g}]"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_records(raw: str) -> list[Any]:
        """Parse a user message into a list of records: a JSON list, a JSON scalar/object, else [None]."""
        if not raw:
            return [None]
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return [None]
        if isinstance(obj, list):
            return obj if obj else [None]
        return [obj]

    # --- mixle distribution/decision capabilities ---
    async def predict(self, records: list[Any], **opts: Any) -> Any:
        quantiles = opts.get("quantiles")
        kwargs: dict[str, Any] = {}
        if quantiles is not None:
            kwargs["quantiles"] = quantiles
        for key in ("interval_level", "n_ensemble", "seed"):
            if key in opts and opts[key] is not None:
                kwargs[key] = opts[key]
        preds = predictive_batch(self._model, records, **kwargs)
        return preds.as_dict()

    async def score(self, records: list[Any], **opts: Any) -> Any:
        recs = list(records)
        if self._service is not None:  # prefer the Service (logs the computation)
            lp = np.asarray(self._service.score(recs), dtype=float)
        else:
            lp = self._log_densities(recs)
        finite = np.isfinite(lp)
        return {
            "log_density": [None if not np.isfinite(v) else float(v) for v in lp],
            "mean_log_density": float(lp[finite].mean()) if finite.any() else None,
            "n_unscorable": int((~finite).sum()),
            "density_semantics": self._density_semantics(),
        }

    def _log_densities(self, recs: list[Any]) -> np.ndarray:
        m = self._model
        if callable(getattr(m, "dist_to_encoder", None)) and callable(getattr(m, "seq_log_density", None)):
            try:
                enc = m.dist_to_encoder().seq_encode(recs)
                return np.asarray(m.seq_log_density(enc), dtype=float)
            except Exception:
                pass
        if not callable(getattr(m, "log_density", None)):
            raise CapabilityError(self._name, "score")
        out = []
        for r in recs:
            try:
                out.append(float(m.log_density(r)))
            except Exception:
                out.append(float("-inf"))
        return np.asarray(out, dtype=float)

    async def latent(self, records: list[Any], **opts: Any) -> Any:
        m = self._model
        recs = list(records)
        if cap.supports(m, cap.LatentStructured):
            lp = m.latent_posterior(recs)
            marg = np.asarray(lp.marginals(), dtype=float)
            result: dict[str, Any] = {
                "kind": type(lp).__name__,
                "marginals": marg.tolist(),
            }
            if callable(getattr(lp, "mode", None)):
                result["mode"] = np.asarray(lp.mode()).tolist()
            if callable(getattr(lp, "entropy", None)):
                result["entropy"] = np.asarray(lp.entropy()).tolist()
            return result
        if callable(getattr(m, "viterbi", None)):  # HMM-style most-probable path
            return {"kind": "viterbi", "path": np.asarray(m.viterbi(recs)).tolist()}
        raise CapabilityError(self._name, "latent")

    async def decide(self, records: list[Any], **opts: Any) -> Any:
        """Bayes-optimal action under a caller-supplied loss + a tail-risk profile.

        ``opts`` must carry ``loss`` (a ``loss(action, draw) -> float`` callable) and ``actions`` (the
        candidate set). ``over`` selects the posterior: ``'predictive'`` (default, draws of new data),
        ``'params'`` (needs retained ``fit_data``), or ``'latent'`` (needs records + a latent model).
        """
        from mixle.inference import posterior as make_posterior

        loss = opts.get("loss")
        actions = opts.get("actions")
        if loss is None or actions is None:
            raise CapabilityError(self._name, "decide")  # decide is meaningless without a loss/actions
        over = opts.get("over", "predictive")
        if over == "params":
            if self._fit_data is None:
                raise CapabilityError(self._name, "decide:params")
            post = make_posterior(self._model, self._fit_data, over="params")
        elif over == "latent":
            post = make_posterior(self._model, list(records), over="latent")
        else:
            post = make_posterior(self._model, over="predictive")
        return bayes_action(
            post,
            loss,
            actions,
            n=opts.get("n", 2000),
            seed=opts.get("seed", 0),
            cvar_alpha=opts.get("cvar_alpha", 0.1),
        )

    def _density_semantics(self) -> str:
        fn = getattr(self._model, "density_semantics", None)
        if callable(fn):
            try:
                return fn().value
            except Exception:
                return "unknown"
        return "exact"


def register_demo_mixle_model(registry: Any, *, name: str = "demo-mixle") -> MixleAdapter:
    """Fit a tiny real 2-component Gaussian mixture and register a :class:`MixleAdapter` for it.

    Self-contained: needs no external data or backend. The fitted mixture supports ``predict`` (ensemble
    path -- a mixture has no closed-form CDF), ``score`` (exact log-density), ``latent`` (component
    responsibilities), ``decide`` (predictive posterior), and ``chat``.
    """
    from mixle.inference import best_of
    from mixle.stats.latent.mixture import MixtureDistribution
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    rng = np.random.RandomState(0)
    data = list(rng.normal(-3.0, 1.0, size=150)) + list(rng.normal(3.0, 1.0, size=150))
    init = MixtureDistribution([GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
    # best-of restarts avoids EM's label-collapse local optima so the demo reliably separates the clusters.
    import io

    _ll, model = best_of(
        data,
        data,
        init.estimator(),
        trials=5,
        max_its=60,
        init_p=0.1,
        delta=1e-8,
        rng=np.random.RandomState(0),
        out=io.StringIO(),  # keep EM iteration logs out of server stdout
    )
    adapter = MixleAdapter(name, model=model, fit_data=data)
    registry.register(adapter)
    return adapter
