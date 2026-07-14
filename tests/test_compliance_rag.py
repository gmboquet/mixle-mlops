"""G8 -- compliance RAG: a fixture permit's discharge limits are extracted into typed, page-cited
``PermitObligation`` records, linked to the monitoring series they govern, and the resulting substrate
lineage is provably intact end to end.
"""

from __future__ import annotations

import time

import pytest
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.governance import Governance
from mixle.substrate.trust import audit_substrate
from mixle.task.trace_record import STEP_KEYS, TRACE_KEYS
from mixle_knowledge.contracts import PermitObligation, SourceRef

from mixle_mlops.multimodal.store import LocalBlobStore
from mixle_mlops.rag import compliance


# --- a tiny, hand-built multi-page PDF (no reportlab dependency) so extraction runs against real bytes,
# not a pre-chunked fixture; pypdf reads real per-page text back out of it. ---------------------------


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf(pages: list[str]) -> bytes:
    n_pages = len(pages)
    catalog_num = 1
    pages_num = 2
    font_num = 3
    page_nums = list(range(4, 4 + 2 * n_pages, 2))
    content_nums = list(range(5, 5 + 2 * n_pages, 2))

    objs: dict[int, bytes] = {
        catalog_num: f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode(),
        pages_num: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_nums)}] /Count {n_pages} >>".encode(),
        font_num: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(pages):
        pnum, cnum = page_nums[i], content_nums[i]
        objs[pnum] = (
            f"<< /Type /Page /Parent {pages_num} 0 R /Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {cnum} 0 R >>"
        ).encode()
        parts = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for j, line in enumerate(text.split("\n")):
            parts.append(f"({_pdf_escape(line)}) Tj" if j == 0 else f"T*\n({_pdf_escape(line)}) Tj")
        stream_body = "\n".join(parts + ["ET"]).encode()
        objs[cnum] = f"<< /Length {len(stream_body)} >>\nstream\n".encode() + stream_body + b"\nendstream"

    max_num = max(objs)
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (max_num + 1)
    for num in range(1, max_num + 1):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {max_num + 1}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, max_num + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {max_num + 1} /Root {catalog_num} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return bytes(out)


_PERMIT_PAGES = [
    "NPDES PERMIT NO. NY-0012345\n"
    "Section 3: Effluent Limitations\n"
    "Total Suspended Solids shall not exceed 30 mg/L, monitored monthly.",
    "pH shall not exceed 9.0 SU, monitored continuously.\nAmmonia shall not exceed 4.5 mg/L, monitored quarterly.",
]


@pytest.fixture
def blob_store(tmp_path):
    return LocalBlobStore(root=tmp_path / "blobs")


@pytest.fixture
def permit_blob_id(blob_store):
    data = _build_pdf(_PERMIT_PAGES)
    record = blob_store.put(data, filename="permit-ny-0012345.pdf", content_type="application/pdf")
    return record.id


def _monitoring_index_with(*, parameter: str, location: str = "outfall-001") -> tuple[Substrate, str]:
    idx = Substrate()
    item = SubstrateItem(
        kind="record",
        text=f"{parameter} monitoring series at {location}",
        payload={"record_type": compliance.MONITORING_SERIES_KIND, "parameter": parameter, "location": location},
        provenance={"source": "monitoring_network"},
    )
    return idx, idx.put(item)


# --- the DoD test -------------------------------------------------------------------------------------


def test_permit_obligations_extracted_with_page_citations_and_linked_and_audited(permit_blob_id, blob_store):
    obligations = compliance.extract_obligations(permit_blob_id, store=blob_store)

    # Three obligations, each typed and traced back to the exact page it was read from.
    by_param = {ob.parameter: ob for ob in obligations}
    assert set(by_param) == {"Total Suspended Solids", "pH", "Ammonia"}
    for ob in obligations:
        assert isinstance(ob, PermitObligation)
        assert ob.permit_id == "NY-0012345"

    tss = by_param["Total Suspended Solids"]
    assert tss.limit == 30.0 and tss.units == "mg/L" and tss.frequency == "monthly"
    assert tss.source.uri.endswith("#page=1")  # page citation: page 1 of the fixture permit
    assert tss.source.sha256 is not None

    ph = by_param["pH"]
    ammonia = by_param["Ammonia"]
    assert ph.source.uri.endswith("#page=2") and ammonia.source.uri.endswith("#page=2")
    assert ph.limit == 9.0 and ammonia.limit == 4.5

    # Link to a named monitoring series ("Total Suspended Solids" at a known outfall) and audit the lineage.
    monitoring_index, series_item_id = _monitoring_index_with(parameter="Total Suspended Solids")
    report = compliance.link_and_audit(obligations, monitoring_index)

    assert report.intact is True
    assert report.dangling == []
    assert tss.monitoring_series_ref == series_item_id  # linked in place to the named monitoring series

    # audit_substrate resolves the same link to a hashed lineage edge across the whole store.
    sweep = audit_substrate(monitoring_index)
    assert sweep["n_broken"] == 0
    assert sweep["n_items"] >= len(obligations) + 2  # obligations + batch item + monitoring series


def test_unmatched_obligation_still_ingests_without_a_dangling_link(permit_blob_id, blob_store):
    """A parameter with no monitoring series yet still gets a provenanced item; it just has no lineage
    edge to a series (an honest gap, not a fabricated link or a crash)."""
    obligations = compliance.extract_obligations(permit_blob_id, store=blob_store)
    empty_index = Substrate()

    report = compliance.link_and_audit(obligations, empty_index)

    assert report.intact is True  # no links at all still resolves (there is nothing to dangle)
    ph = next(ob for ob in obligations if ob.parameter == "pH")
    assert ph.monitoring_series_ref is None


def test_report_is_an_ic5_envelope_and_gates_publication_behind_governance(permit_blob_id, blob_store):
    obligations = compliance.extract_obligations(permit_blob_id, store=blob_store)
    monitoring_index, _series_item_id = _monitoring_index_with(parameter="Total Suspended Solids")
    compliance.link_and_audit(obligations, monitoring_index)

    tss = next(ob for ob in obligations if ob.parameter == "Total Suspended Solids")
    tss_status = compliance.obligation_status(tss, observed_value=22.0, last_monitored_at=time.time())
    assert tss_status == "met"

    obligation_item_ids = [item.id for item in monitoring_index.all(kind="record") if "obligation" in item.payload]
    statuses = {obligation_item_ids[0]: tss_status}
    report = compliance.build_compliance_report(obligations, obligation_item_ids, statuses=statuses)

    assert set(TRACE_KEYS) <= set(report)
    for step in report["steps"]:
        assert set(STEP_KEYS) <= set(step)

    governance = Governance().grant("regulator-1", "compliance-published")
    results = compliance.publish_report(
        monitoring_index, obligation_item_ids, to="compliance-published", by="regulator-1", governance=governance
    )
    assert results and all(results)

    # An unauthorized approver cannot promote a report -- governance actually gates it.
    other_index, _ = _monitoring_index_with(parameter="Ammonia")
    compliance.link_and_audit(obligations, other_index)
    other_ids = [item.id for item in other_index.all(kind="record") if "obligation" in item.payload]
    denied = compliance.publish_report(
        other_index, other_ids, to="compliance-published", by="someone-else", governance=governance
    )
    assert denied and not any(denied)


def test_obligation_status_flags_exceeded_and_overdue():
    obligation = PermitObligation(
        permit_id="NY-0012345",
        parameter="pH",
        limit=9.0,
        units="SU",
        frequency="continuous",
        monitoring_series_ref=None,
        source=SourceRef(uri="mixle://document/permit-ny-0012345#page=2"),
    )

    assert compliance.obligation_status(obligation, observed_value=9.4) == "exceeded"
    assert compliance.obligation_status(obligation, observed_value=8.5, last_monitored_at=time.time()) == "met"
    assert compliance.obligation_status(obligation, last_monitored_at=time.time() - 10 * 24 * 3600) == "overdue"
    assert compliance.obligation_status(obligation) == "overdue"  # never monitored
