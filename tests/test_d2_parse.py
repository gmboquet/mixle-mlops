"""D2 -- RAG parsing: OCR, tables, LAS; page-level provenance.

Covers:
  * a 2-page image-only PDF: pypdf sees no text layer, ``parse_and_chunk_located`` OCR-falls-back per page and
    tags every chunk with ``page`` + an ``artifact_ref``/``selector`` that hydrates back to the exact original
    bytes;
  * a 3-row assay XLSX: ``extract_typed_tables`` builds one IC-13 ``KnowledgeItem`` (typed-table schema) whose
    payload preserves column order/types/units and exact cell values -- including the null-vs-empty-string
    distinction -- and whose ``content_hash`` is unaffected by the text surface; ``parse_and_chunk_located``
    emits one chunk per row carrying the same artifact ref;
  * a retrieval hit (via the real index → embed → retrieve path) carries that page/row + artifact ref/selector
    through ``meta``;
  * a well-log LAS file parses into curves/units/depth via ``lasio``.
"""
from __future__ import annotations

import io
import zipfile

import mixle_mlops.rag.embeddings as emb_mod
import mixle_mlops.rag.vectorstore as vs_mod
import openpyxl
import pytest

from mixle_mlops.documents import structured_artifacts as sa
from mixle_mlops.documents.parse import (
    DocumentParseError,
    extract_text,
    extract_typed_tables,
    parse_and_chunk_located,
)
from mixle_mlops.documents.structured_artifacts import TYPED_TABLE_SCHEMA
from mixle_mlops.multimodal.store import LocalBlobStore
from mixle_mlops.rag.embeddings import Embedder
from mixle_mlops.rag.index import index_document_chunks, retrieve
from mixle_mlops.rag.vectorstore import LocalVectorStore


# --- fixtures: build the raw bytes for a scanned PDF and an assay XLSX --------------------------------------

def _make_image_only_pdf(page_texts: list[str]) -> bytes:
    """A PDF with zero text layer -- each page is a rendered image of ``page_texts[i]``, pypdf sees nothing."""
    from PIL import Image, ImageDraw

    def _page(text: str):
        img = Image.new("RGB", (400, 200), "white")
        ImageDraw.Draw(img).text((20, 80), text, fill="black")
        return img

    pages = [_page(t) for t in page_texts]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


def _make_assay_xlsx() -> bytes:
    """A 3-row assay sheet: ``*SampleID`` (declared pk), ``Depth [m]``, ``Grade [g/t]``, ``Notes``.

    Row 2's ``Notes`` is a real empty string, row 3's is blank (``None``) -- openpyxl's own ``Workbook.save``
    collapses an assigned ``""`` down to an empty cell (an established library limitation: its writer special-
    cases ``value == ""`` identically to ``None``), so we build normally and then patch the one cell's raw XML
    to the inline-string-with-empty-text form the *reader* correctly resolves back to ``""``.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Assay"
    ws.append(["*SampleID", "Depth [m]", "Grade [g/t]", "Notes"])
    ws.append([1, 10.5, 2.3, ""])              # Notes: empty string (patched in below)
    ws.append([2, 12.0, None, "visible vein"])  # Grade: null
    ws.append([3, 13.25, 0.0, None])            # Notes: null (genuinely blank)
    buf = io.BytesIO()
    wb.save(buf)
    data = buf.getvalue()

    zin = zipfile.ZipFile(io.BytesIO(data))
    sheet_name = next(n for n in zin.namelist() if n.startswith("xl/worksheets/sheet"))
    xml = zin.read(sheet_name).decode()
    patched = xml.replace('<c r="D2" t="inlineStr"></c>', '<c r="D2" t="inlineStr"><is><t></t></is></c>')
    assert patched != xml, "fixture assumption about openpyxl's empty-cell XML shape broke"
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zout:
        for item in zin.infolist():
            content = zin.read(item.filename)
            if item.filename == sheet_name:
                content = patched.encode()
            zout.writestr(item, content)
    return out.getvalue()


def _make_las() -> bytes:
    return (
        "~VERSION INFORMATION\n VERS.   2.0 :\n WRAP.   NO  :\n"
        "~WELL INFORMATION\n"
        " STRT.M    100.0 :\n STOP.M    102.0 :\n STEP.M    1.0 :\n"
        " NULL.     -999.25 :\n WELL.     TEST-1 :\n"
        "~CURVE INFORMATION\n DEPT.M   : Depth\n GR  .GAPI: Gamma Ray\n"
        "~ASCII\n"
        "100.0  50.1\n101.0  55.2\n102.0  -999.25\n"
    ).encode("ascii")


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def local_embedder(monkeypatch):
    """Deterministic local embedder -- no embeddings server needed for the retrieval assertions."""
    emb_mod.reset_embedder()
    monkeypatch.setattr(emb_mod, "get_embedder", lambda: Embedder(allow_remote=False))
    import mixle_mlops.rag.index as index_mod

    monkeypatch.setattr(index_mod, "get_embedder", lambda: Embedder(allow_remote=False))
    yield
    emb_mod.reset_embedder()


@pytest.fixture
def vector_store(tmp_path, monkeypatch):
    import mixle_mlops.storage.db as db

    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    db._engine = None
    vs_mod.reset_vector_store()
    yield LocalVectorStore()
    vs_mod.reset_vector_store()
    db._engine = None


# --- PDF: OCR fallback + page provenance ---------------------------------------------------------------------

def test_image_only_pdf_ocr_falls_back_per_page(store, monkeypatch):
    import mixle_mlops.documents.parse as parse_mod

    monkeypatch.setattr(parse_mod, "get_blob_store", lambda: store)

    data = _make_image_only_pdf(["HELLO PAGE ONE", "SECOND PAGE TEXT"])

    # the text layer is genuinely empty -- pypdf alone would see nothing.
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    assert all((p.extract_text() or "").strip() == "" for p in reader.pages)

    # extract_text() OCR-falls-back and recovers both pages' words.
    text = extract_text(data, filename="scan.pdf", content_type="application/pdf")
    assert "HELLO" in text.upper() and "SECOND PAGE" in text.upper()

    located = parse_and_chunk_located(data, filename="scan.pdf", content_type="application/pdf")
    assert len(located) == 2                              # one short OCR'd chunk per page
    pages = {c.page for c in located}
    assert pages == {0, 1}
    for c in located:
        assert c.source_format == "pdf"
        assert c.row is None and c.col is None
        assert c.artifact_ref
        assert c.selector == {"page": c.page}
    page0 = next(c for c in located if c.page == 0)
    page1 = next(c for c in located if c.page == 1)
    assert "HELLO" in page0.text.upper()
    assert "SECOND PAGE" in page1.text.upper()

    # hydrating the artifact ref points at the exact original bytes.
    assert sa.load_artifact_bytes(page0.artifact_ref, store) == data
    assert page0.artifact_ref == page1.artifact_ref        # both pages share one source-document artifact


def test_pdf_ocr_unavailable_degrades_gracefully(monkeypatch):
    """Without the OCR extra, a scanned page just yields no text -- extract_text never raises."""
    import mixle_mlops.documents.parse as parse_mod

    def _boom(*a, **k):
        raise DocumentParseError("ocr extra missing")

    monkeypatch.setattr(parse_mod, "_extract_pdf_ocr", _boom)
    data = _make_image_only_pdf(["ANYTHING"])
    assert extract_text(data, filename="scan.pdf") == ""


# --- XLSX: typed table + row-level provenance ------------------------------------------------------------

def test_assay_xlsx_typed_table_round_trips_types_units_and_nulls(store):
    data = _make_assay_xlsx()

    items = extract_typed_tables(data, filename="assay.xlsx", store=store)
    assert len(items) == 1
    item = items[0]

    assert item["schema_uri"] == TYPED_TABLE_SCHEMA
    assert len(item["content_hash"]) == 64
    assert item["artifact_ref"]
    assert item["metadata"]["sheet_name"] == "Assay"
    assert any(r.get("metadata", {}).get("sheet_name") == "Assay" for r in item["relations"])

    payload = item["payload"]
    assert payload["primary_key"] == ["SampleID"]
    names = [c["name"] for c in payload["columns"]]
    assert names == ["SampleID", "Depth", "Grade", "Notes"]   # column order preserved

    by_name = {c["name"]: c for c in payload["columns"]}
    assert by_name["SampleID"]["type"] == "integer" and by_name["SampleID"]["nullable"] is False
    assert by_name["Depth"]["type"] == "float" and by_name["Depth"]["unit"] == "m"
    assert by_name["Grade"]["type"] == "float" and by_name["Grade"]["unit"] == "g/t"
    assert by_name["Grade"]["nullable"] is True             # row 2 has a null Grade
    assert by_name["Notes"]["type"] == "string" and by_name["Notes"]["unit"] is None

    rows = payload["rows"]
    assert len(rows) == 3
    by_id = {r["SampleID"]: r for r in rows}
    assert by_id[1] == {"SampleID": 1, "Depth": 10.5, "Grade": 2.3, "Notes": ""}     # empty string, not null
    assert by_id[2]["Grade"] is None                        # a genuinely missing numeric cell stays null
    assert by_id[3]["Notes"] is None                        # a genuinely blank cell stays null, not ""
    assert by_id[1]["Notes"] is not None                    # the null/"" distinction actually holds

    # hydrating the workbook artifact reproduces the exact original bytes.
    assert sa.load_artifact_bytes(item["artifact_ref"], store) == data

    # deleting the text surface never changes the table's identity.
    stripped = dict(item)
    stripped["text_surface"] = None
    assert sa.content_hash_payload(stripped["payload"]) == item["content_hash"]
    reextracted = extract_typed_tables(data, filename="assay.xlsx", store=store)
    assert reextracted[0]["content_hash"] == item["content_hash"]
    assert reextracted[0]["payload"] == item["payload"]


def test_xlsx_located_chunks_carry_row_and_shared_artifact_ref(store, monkeypatch):
    """One ``parse_and_chunk_located`` pass (the real ingestion path) persists the workbook exactly once, so
    every row-chunk and the typed-table item built alongside it share one ``artifact_ref``."""
    import mixle_mlops.documents.parse as parse_mod

    monkeypatch.setattr(parse_mod, "get_blob_store", lambda: store)
    data = _make_assay_xlsx()

    located = parse_and_chunk_located(data, filename="assay.xlsx")

    assert len(located) == 3
    rows_seen = sorted(c.row for c in located)
    assert rows_seen == [0, 1, 2]
    shared_ref = located[0].artifact_ref
    for c in located:
        assert c.page is None and c.col is None
        assert c.source_format == "xlsx"
        assert c.artifact_ref == shared_ref                 # same workbook artifact across every row
        assert c.selector["sheet"] == "Assay"
        assert c.selector["row_start"] == c.selector["row_end"] == c.row
        assert c.selector["columns"] == ["SampleID", "Depth", "Grade", "Notes"]
    row0 = next(c for c in located if c.row == 0)
    assert "SampleID=1" in row0.text and "Depth=10.5" in row0.text     # "col=val" display surface
    assert sa.load_artifact_bytes(shared_ref, store) == data


# --- retrieval: a hit carries page/row + artifact ref/selector ------------------------------------------------

def test_retrieval_hit_carries_page_and_artifact_provenance(store, local_embedder, vector_store, monkeypatch):
    import mixle_mlops.documents.parse as parse_mod

    monkeypatch.setattr(parse_mod, "get_blob_store", lambda: store)

    pdf_bytes = _make_image_only_pdf(["QUARTZ VEIN OBSERVED HERE"])
    located = parse_and_chunk_located(pdf_bytes, filename="scan.pdf", content_type="application/pdf")
    assert located

    for i, chunk in enumerate(located):
        extra_meta = {
            "page": chunk.page, "row": chunk.row, "col": chunk.col,
            "source_format": chunk.source_format, "artifact_ref": chunk.artifact_ref,
            "selector": chunk.selector,
        }
        index_document_chunks(
            "user-1", "doc-pdf", [chunk.text], filename="scan.pdf",
            extra_meta=extra_meta, store=vector_store, replace=(i == 0),
        )

    hits = retrieve("user-1", "quartz vein", k=3, store=vector_store)
    assert hits
    hit = hits[0]
    assert hit["meta"]["page"] == 0
    assert hit["meta"]["artifact_ref"] == located[0].artifact_ref
    assert hit["meta"]["selector"] == {"page": 0}
    assert sa.load_artifact_bytes(hit["meta"]["artifact_ref"], store) == pdf_bytes


def test_retrieval_hit_carries_row_and_artifact_provenance(store, local_embedder, vector_store, monkeypatch):
    import mixle_mlops.documents.parse as parse_mod

    monkeypatch.setattr(parse_mod, "get_blob_store", lambda: store)

    xlsx_bytes = _make_assay_xlsx()
    located = parse_and_chunk_located(xlsx_bytes, filename="assay.xlsx")
    assert located

    for i, chunk in enumerate(located):
        extra_meta = {
            "page": chunk.page, "row": chunk.row, "col": chunk.col,
            "source_format": chunk.source_format, "artifact_ref": chunk.artifact_ref,
            "selector": chunk.selector,
        }
        index_document_chunks(
            "user-1", "doc-xlsx", [chunk.text], filename="assay.xlsx",
            extra_meta=extra_meta, store=vector_store, replace=(i == 0),
        )

    hits = retrieve("user-1", "SampleID=2 Depth=12.0", k=3, store=vector_store)
    assert hits
    hit = next(h for h in hits if h["meta"]["row"] == 1)
    assert hit["meta"]["source_format"] == "xlsx"
    assert hit["meta"]["artifact_ref"] == located[0].artifact_ref
    assert hit["meta"]["selector"]["sheet"] == "Assay"
    assert sa.load_artifact_bytes(hit["meta"]["artifact_ref"], store) == xlsx_bytes


# --- LAS: curves/units/depth -----------------------------------------------------------------------------

def test_las_well_log_parses_curves_units_and_depth(store, monkeypatch):
    import mixle_mlops.documents.parse as parse_mod

    monkeypatch.setattr(parse_mod, "get_blob_store", lambda: store)
    data = _make_las()

    text = extract_text(data, filename="well.las")
    assert "GR" in text and "GAPI" in text

    located = parse_and_chunk_located(data, filename="well.las")
    assert located
    c = located[0]
    assert c.source_format == "las" and c.page is None and c.row is None
    assert c.artifact_ref
    assert c.selector["curves"] == ["DEPT", "GR"]
    assert c.selector["depth_start"] == 100.0 and c.selector["depth_stop"] == 102.0
    assert sa.load_artifact_bytes(c.artifact_ref, store) == data
