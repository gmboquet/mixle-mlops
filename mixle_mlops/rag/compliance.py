"""Compliance RAG: permit/obligation extraction, lineage tracking, and audit trail (G8).

A regulator's permit states discharge limits and monitoring obligations in prose. This module turns
that prose into typed, page-cited :class:`~mixle_knowledge.contracts.PermitObligation` records
(:func:`extract_obligations`) and persists each one as a provenanced ``mixle.substrate`` item linked to
the monitoring series it governs (:func:`link_and_audit`) -- so an environmental report never states a
number without a citation a reviewer can jump to, and a lineage sweep (``verify_lineage`` /
``audit_substrate``) can prove the whole chain still resolves.

Building blocks, each reused rather than reinvented:

  * page-preserving text extraction -- like D2's :mod:`mixle_mlops.documents.parse`, but keeping every
    page separately addressable (D2's ``extract_text`` joins PDF pages into one blob, which loses exactly
    the page citation a permit obligation needs).
  * ``mixle.substrate`` -- :class:`~mixle.substrate.core.SubstrateItem` for storage,
    :func:`~mixle.substrate.trust.verify_lineage` / :func:`~mixle.substrate.trust.audit_substrate` for the
    lineage sweep, :func:`~mixle.substrate.governance.propose` / :func:`~mixle.substrate.governance.approve`
    to gate publication.
  * :mod:`mixle.task.trace_record` (IC-5) -- every compliance report is validated against the frozen
    ``{prompt, steps, outcome, provenance}`` envelope before it may be proposed for approval.
  * :mod:`mixle_knowledge.contracts` -- ``PermitObligation`` is the typed regulatory contract (owned by
    mixle-knowledge per DR-OWN); imported lazily here, matching the convention this repo already uses for
    IC-13 knowledge-bundle interop (``mixle_mlops.gateway.tool_registry``) since ``mixle_knowledge`` is not
    yet a hard dependency of this package.

Extraction is deliberately NOT an LLM prose-trust exercise: :func:`_parse_obligation_lines` only ever
emits an obligation when the source page literally states it in a recognisable
"<parameter> shall not exceed <limit> <units>, monitored <frequency>" shape, so every numeric field this
module produces is directly re-derivable from the cited page -- the discipline the work order calls out
("never trust bare LLM prose"). Jurisdiction-specific permit language, OCR/table internals (D2), and
automatic regulatory filing are explicitly out of scope (non-goals).
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from mixle.data.hashing import dataset_hash
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.governance import Governance, approve, propose
from mixle.substrate.trust import LineageReport, verify_lineage
from mixle.task.trace_record import TraceRecord, validate_trace_record

from ..multimodal.store import BlobStore, get_blob_store

if TYPE_CHECKING:  # pragma: no cover - typing only; the real import is lazy (see module docstring)
    from mixle_knowledge.contracts import PermitObligation

__all__ = [
    "MONITORING_SERIES_KIND",
    "extract_obligations",
    "link_and_audit",
    "build_compliance_report",
    "publish_report",
    "obligation_status",
]

MONITORING_SERIES_KIND = "monitoring_series"  # a payload["record_type"] marker, not a SubstrateItem.kind
# mixle.substrate.core.MODALITIES is a frozen, domain-agnostic list of SubstrateItem kinds with no
# "monitoring_series"/"obligation" entries -- extending it is out of this task's scope. Every item this
# module reads or writes therefore lives under the generic "record" modality and is distinguished by
# ``payload["record_type"]`` instead (``MONITORING_SERIES_KIND`` for a monitoring series, "permit_obligation"
# for an obligation, "compliance_link_batch" for the audit batch).
_SUBSTRATE_KIND = "record"
_OBLIGATION_KIND = _SUBSTRATE_KIND

# --- deterministic, page-grounded obligation extraction (never a bare LLM claim) --------------------

_OBLIGATION_LINE_RE = re.compile(
    r"^(?P<parameter>[A-Za-z][A-Za-z0-9 /()\-]*?)\s+shall not exceed\s+"
    r"(?P<limit>\d+(?:\.\d+)?)\s*(?P<units>[A-Za-z/%°µ]+)\s*[,;]?\s*"
    r"monitored\s+(?P<frequency>[A-Za-z]+)\.?\s*$",
    re.IGNORECASE,
)
_PERMIT_ID_RE = re.compile(r"PERMIT\s+NO\.?\s*[:\-]?\s*([A-Za-z0-9\-]+)", re.IGNORECASE)


def _is_pdf(filename: str, content_type: str | None) -> bool:
    ext = PurePosixPath(filename or "").suffix.lower().lstrip(".")
    if ext == "pdf":
        return True
    return (content_type or "").split(";", 1)[0].strip().lower() == "application/pdf"


def _extract_pages(data: bytes, *, filename: str = "", content_type: str | None = None) -> list[str]:
    """Page-preserving text extraction. PDFs keep their real page boundaries via ``pypdf`` (lazy import,
    same optional dependency D2's parser already uses); plain text/markdown splits on a form-feed page
    marker (``\\f``) when present, else the whole document is page 1."""
    if _is_pdf(filename, content_type):
        try:
            import pypdf  # lazy: documents extra (mirrors mixle_mlops.documents.parse._extract_pdf)
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "PDF permit parsing needs the 'documents' extra (pip install mixle-mlops[documents])."
            ) from exc
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - a single unreadable page shouldn't sink the whole permit
                pages.append("")
        return pages
    text = data.decode("utf-8", errors="replace")
    return text.split("\f") if "\f" in text else [text]


def _parse_obligation_lines(page_text: str) -> list[dict[str, str]]:
    """Regex-grounded obligation parsing over one page's text. A row is only emitted when a line matches
    the recognised shape verbatim, so it is always re-derivable from ``page_text`` -- the anti-hallucination
    discipline the work order calls out."""
    found: list[dict[str, str]] = []
    for line in page_text.splitlines():
        m = _OBLIGATION_LINE_RE.match(line.strip())
        if m:
            found.append(
                {
                    "parameter": " ".join(m.group("parameter").split()),
                    "limit": m.group("limit"),
                    "units": m.group("units"),
                    "frequency": m.group("frequency").lower(),
                }
            )
    return found


def _guess_permit_id(pages: Iterable[str]) -> str | None:
    for page in pages:
        m = _PERMIT_ID_RE.search(page)
        if m:
            return m.group(1)
    return None


def extract_obligations(
    permit_doc_ref: str,
    *,
    store: BlobStore | None = None,
    uri: str | None = None,
) -> list["PermitObligation"]:
    """Parse a permit/regulatory document into typed :class:`PermitObligation` records.

    ``permit_doc_ref`` is a blob id from the platform's :class:`~mixle_mlops.multimodal.store.BlobStore`
    -- the same store D2's document upload path (``mixle_mlops.gateway.routes.rag.upload_document``)
    already writes into, so a permit is just another uploaded document. Every obligation's ``source`` cites
    the exact page it was read from (``{uri}#page={n}``) plus the sha256 of the whole document, so a
    reviewer -- or a re-run of this function -- can verify the claim against the source bytes directly.
    """
    from mixle_knowledge.contracts import PermitObligation, SourceRef

    store = store or get_blob_store()
    record, data = store.get(permit_doc_ref)
    pages = _extract_pages(data, filename=record.filename, content_type=record.content_type)
    base_uri = uri or f"blob://{record.id}"
    doc_hash = hashlib.sha256(data).hexdigest()
    permit_id = _guess_permit_id(pages) or record.filename or record.id

    obligations: list[PermitObligation] = []
    for page_no, page_text in enumerate(pages, start=1):
        for row in _parse_obligation_lines(page_text):
            obligations.append(
                PermitObligation(
                    permit_id=permit_id,
                    parameter=row["parameter"],
                    limit=float(row["limit"]),
                    units=row["units"],
                    frequency=row["frequency"],
                    monitoring_series_ref=None,
                    source=SourceRef(uri=f"{base_uri}#page={page_no}", sha256=doc_hash),
                )
            )
    return obligations


# --- substrate lineage: obligation <-> monitoring series, provenanced -------------------------------


def _obligation_hash(ob: "PermitObligation") -> str:
    """Content hash of an obligation's declared fields (the reusable ``mixle.data.hashing.dataset_hash``
    utility E7's provenance chain already relies on) -- re-derivable from the record alone, independent of
    where it happens to be stored."""
    return dataset_hash(
        [
            {
                "permit_id": ob.permit_id,
                "parameter": ob.parameter,
                "limit": ob.limit,
                "units": ob.units,
                "frequency": ob.frequency,
                "source_uri": ob.source.uri,
            }
        ]
    )


def _is_monitoring_series(item: SubstrateItem) -> bool:
    return item.kind == _SUBSTRATE_KIND and item.payload.get("record_type") == MONITORING_SERIES_KIND


def _match_monitoring_series(
    monitoring_index: Substrate, *, parameter: str, location: str | None = None
) -> SubstrateItem | None:
    """D3-style metadata match, applied to the substrate instead of the vector store: a monitoring series
    governs an obligation when its declared ``parameter`` matches (case-insensitively) and -- when the
    obligation carries a ``location`` -- its declared ``location`` matches too. Falls back to
    ``Substrate.search`` (lexical/embedding) so an unusually-named series can still be found by its text."""
    target = parameter.strip().lower()
    candidates = [item for item in monitoring_index.all(kind=_SUBSTRATE_KIND) if _is_monitoring_series(item)]
    for item in candidates:
        series_param = str(item.payload.get("parameter", "")).strip().lower()
        if series_param != target:
            continue
        if location is None:
            return item
        if str(item.payload.get("location", "")).strip().lower() == location.strip().lower():
            return item
    hits = monitoring_index.search(parameter, k=len(candidates) or 1)
    for hit_item, _score in hits:
        if _is_monitoring_series(hit_item):
            return hit_item
    return None


def link_and_audit(
    obligations: Sequence["PermitObligation"],
    monitoring_index: Substrate,
    *,
    substrate: Substrate | None = None,
    location: str | None = None,
) -> LineageReport:
    """Ingest every obligation as a provenanced :class:`~mixle.substrate.core.SubstrateItem`, link it to
    the monitoring series it governs, and audit the resulting lineage graph in one pass.

    Each obligation becomes its own substrate item (``provenance.content_hash`` = :func:`_obligation_hash`,
    re-derivable from the record alone) whose ``links`` point at the matched monitoring-series item. A
    single batch item is then created linking to every obligation item, and
    :func:`~mixle.substrate.trust.verify_lineage` walks it: the returned :class:`LineageReport` is
    ``intact`` iff every obligation resolves to an existing monitoring series, transitively, in one call.

    Matched obligations have their ``monitoring_series_ref`` filled in place, so the caller's records
    reflect the same linkage the substrate now carries.
    """
    substrate = substrate if substrate is not None else monitoring_index
    obligation_item_ids: list[str] = []
    for ob in obligations:
        series_item = _match_monitoring_series(monitoring_index, parameter=ob.parameter, location=location)
        links = [series_item.id] if series_item is not None else []
        item = SubstrateItem(
            kind=_OBLIGATION_KIND,
            text=f"{ob.parameter} limit {ob.limit} {ob.units}, monitored {ob.frequency} (permit {ob.permit_id})",
            payload={"record_type": "permit_obligation", "obligation": ob.model_dump(mode="json")},
            provenance={
                "source": "permit_obligation",
                "permit_id": ob.permit_id,
                "content_hash": _obligation_hash(ob),
                "source_uri": ob.source.uri,
            },
            links=links,
            tags=[ob.parameter, ob.permit_id],
        )
        item_id = substrate.put(item)
        obligation_item_ids.append(item_id)
        if series_item is not None:
            ob.monitoring_series_ref = series_item.id

    batch = SubstrateItem(
        kind=_OBLIGATION_KIND,
        text=f"compliance link batch: {len(obligation_item_ids)} obligation(s)",
        payload={"record_type": "compliance_link_batch", "obligation_item_ids": obligation_item_ids},
        provenance={"source": "link_and_audit", "n_obligations": len(obligation_item_ids)},
        links=list(obligation_item_ids),
    )
    batch_id = substrate.put(batch)
    return verify_lineage(substrate, batch_id)


# --- IC-5 report envelope + governance-gated publication --------------------------------------------


def build_compliance_report(
    obligations: Sequence["PermitObligation"],
    obligation_item_ids: Sequence[str],
    *,
    prompt: str = "environmental compliance report",
    statuses: Mapping[str, str] | None = None,
) -> TraceRecord:
    """Build the frozen IC-5 envelope for an environmental report: one step per obligation, each step's
    ``result.content_hash`` re-derivable from the obligation alone. Every reported number resolves to its
    data + obligation edge this way, before governance ever sees the report (Algorithm step 4)."""
    if len(obligations) != len(obligation_item_ids):
        raise ValueError("obligations and obligation_item_ids must be the same length")
    steps = []
    for ob, item_id in zip(obligations, obligation_item_ids):
        steps.append(
            {
                "tool": "extract_obligations",
                "args": {"permit_id": ob.permit_id, "parameter": ob.parameter, "source_uri": ob.source.uri},
                "result": {
                    "obligation_item_id": item_id,
                    "limit": ob.limit,
                    "units": ob.units,
                    "content_hash": _obligation_hash(ob),
                    "status": (statuses or {}).get(item_id),
                },
                "model": None,
                "verdict": None,
            }
        )
    record: TraceRecord = {
        "prompt": prompt,
        "steps": steps,
        "outcome": {"n_obligations": len(obligations), "statuses": dict(statuses or {})},
        "provenance": {"obligation_item_ids": list(obligation_item_ids)},
    }
    validate_trace_record(record)  # the frozen envelope check itself -- never skip it before publishing
    return record


def publish_report(
    substrate: Substrate,
    report_item_ids: Sequence[str],
    *,
    to: str,
    by: str,
    governance: Governance,
) -> list[bool]:
    """Gate publication behind governance approval: propose every item into scope ``to``, then approve
    them (only succeeds when ``by`` is authorized for ``to``). Returns one bool per proposed id (``True``
    iff that item was promoted)."""
    proposed = propose(substrate, list(report_item_ids), to=to, by=by)
    return [approve(substrate, item_id, by=by, governance=governance, to=to) for item_id in proposed]


# --- obligation-status query: joins an obligation to whatever monitoring evidence exists -------------

_FREQUENCY_WINDOW_S: dict[str, float] = {
    "continuous": 1 * 24 * 3600,
    "daily": 2 * 24 * 3600,
    "weekly": 9 * 24 * 3600,
    "monthly": 40 * 24 * 3600,
    "quarterly": 100 * 24 * 3600,
    "semiannual": 200 * 24 * 3600,
    "annual": 400 * 24 * 3600,
    "annually": 400 * 24 * 3600,
}


def _is_overdue(frequency: str, last_monitored_at: float | None, now: float) -> bool:
    if last_monitored_at is None:
        return True
    window = _FREQUENCY_WINDOW_S.get(frequency.strip().lower())
    if window is None:
        return False  # unrecognised cadence: no overdue verdict without a jurisdiction rule (non-goal)
    return (now - last_monitored_at) > window


def obligation_status(
    obligation: "PermitObligation",
    *,
    observed_value: float | None = None,
    exceedance_probability: float | None = None,
    last_monitored_at: float | None = None,
    now: float | None = None,
    alarm_threshold: float = 0.5,
) -> str:
    """Join one obligation to whatever monitoring evidence is available and report
    ``"met"`` / ``"exceeded"`` / ``"overdue"`` / ``"unknown"``.

    ``observed_value`` is a direct measurement; ``exceedance_probability`` is a G7-style ``prob_exceed``
    (IC-8) reading -- G7 is a separate task and not a G8 dependency, so this takes its output as a plain
    probability rather than importing a module that may not have landed. ``last_monitored_at`` plus the
    obligation's declared ``frequency`` flags a report as overdue before an exceedance is even measurable.
    """
    now = time.time() if now is None else float(now)
    overdue = _is_overdue(obligation.frequency, last_monitored_at, now)
    exceeded: bool | None
    if observed_value is not None:
        exceeded = observed_value > obligation.limit
    elif exceedance_probability is not None:
        exceeded = exceedance_probability >= alarm_threshold
    else:
        exceeded = None
    if exceeded:
        return "exceeded"
    if overdue:
        return "overdue"
    if exceeded is False:
        return "met"
    return "unknown"
