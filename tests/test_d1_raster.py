"""D1 — large-raster / GeoTIFF tiling. A synthetic 120-megapixel single-band GeoTIFF (known affine, written
with rasterio) must tile into PNGs small enough for a vision backend, each carrying a ground extent whose
union reconstructs the source bounds, plus a positive ground-scale. Also covers the two additive content.py
edits (the ``image/tiff`` mime + the raised raster byte ceiling) and the blob-store-backed image parts path.
Skips outright if the ``raster`` extra (rasterio/pillow) isn't installed."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mixle_mlops.multimodal.content import (
    ALLOWED_IMAGE_TYPES,
    MAX_IMAGE_BYTES,
    MAX_RASTER_BYTES,
    MultimodalError,
    guard_image,
)
from mixle_mlops.multimodal.store import LocalBlobStore

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rasterio") is None or importlib.util.find_spec("PIL") is None,
    reason="raster extra (rasterio, pillow) not installed",
)

WIDTH, HEIGHT = 12000, 10000  # exactly 120,000,000 pixels
PIXEL_SIZE = 2.5
WEST, NORTH = 500_000.0, 4_500_000.0


def _synthetic_geotiff() -> bytes:
    """A single-band 120 MP GeoTIFF with a known north-up affine transform."""
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    transform = from_origin(WEST, NORTH, PIXEL_SIZE, PIXEL_SIZE)
    # cheap, non-constant band data (a coarse gradient) so the uint8 stretch has real dynamic range
    rows = np.arange(HEIGHT, dtype=np.uint16).reshape(-1, 1)
    cols = np.arange(WIDTH, dtype=np.uint16).reshape(1, -1)
    band = ((rows % 256) + (cols % 256)).astype(np.uint8)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=HEIGHT,
            width=WIDTH,
            count=1,
            dtype="uint8",
            crs="EPSG:32633",
            transform=transform,
        ) as dst:
            dst.write(band, 1)
        return bytes(memfile.read())


def _source_bounds() -> tuple[float, float, float, float]:
    from rasterio.transform import from_origin

    transform = from_origin(WEST, NORTH, PIXEL_SIZE, PIXEL_SIZE)
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (WIDTH, HEIGHT)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def test_tile_raster_120mp_geotiff():
    from mixle_mlops.multimodal.raster import RasterTile, tile_raster

    data = _synthetic_geotiff()
    tiles = tile_raster(data)

    assert len(tiles) >= 1
    for t in tiles:
        assert isinstance(t, RasterTile)
        assert len(t.png) <= MAX_IMAGE_BYTES
        # must not raise: every tile is itself a guard-passing image
        guard_image(content_type="image/png", size=len(t.png))
        # a real PNG (magic bytes), not just arbitrary bytes
        assert t.png[:8] == b"\x89PNG\r\n\x1a\n"

    minx = min(t.extent[0] for t in tiles)
    miny = min(t.extent[1] for t in tiles)
    maxx = max(t.extent[2] for t in tiles)
    maxy = max(t.extent[3] for t in tiles)
    src_minx, src_miny, src_maxx, src_maxy = _source_bounds()

    assert minx == pytest.approx(src_minx, abs=1e-3)
    assert miny == pytest.approx(src_miny, abs=1e-3)
    assert maxx == pytest.approx(src_maxx, abs=1e-3)
    assert maxy == pytest.approx(src_maxy, abs=1e-3)

    assert tiles[0].scale > 0


def test_tile_raster_respects_max_tile_px_and_overlap():
    from mixle_mlops.multimodal.raster import tile_raster

    data = _synthetic_geotiff()
    tiles = tile_raster(data, max_tile_px=512, overlap_px=32, downsample_to_mp=6.0)

    assert len(tiles) > 1
    rows = {t.row for t in tiles}
    cols = {t.col for t in tiles}
    assert len(rows) > 1 and len(cols) > 1
    # every tile still passes the guard at the smaller tile size too
    for t in tiles:
        guard_image(content_type="image/png", size=len(t.png))


def test_raster_to_image_parts_round_trips_through_blob_store(tmp_path):
    from mixle_mlops.multimodal.raster import raster_to_image_parts, tile_raster

    data = _synthetic_geotiff()
    store = LocalBlobStore(tmp_path / "blobs")
    tiles = tile_raster(data)
    parts = raster_to_image_parts(data, store)

    assert len(parts) == len(tiles)
    for part, tile in zip(parts, tiles):
        file_id = part.image_url["file_id"]
        assert store.has(file_id)
        record, stored_bytes = store.get(file_id)
        assert stored_bytes == tile.png
        assert record.content_type == "image/png"


def test_tiff_allowed_with_raised_byte_ceiling():
    assert "image/tiff" in ALLOWED_IMAGE_TYPES
    assert MAX_RASTER_BYTES > MAX_IMAGE_BYTES

    # a raw geotiff-sized upload well above the normal image ceiling, but under the raster ceiling, passes
    guard_image(content_type="image/tiff", size=MAX_IMAGE_BYTES + 1)

    # still bounded: an absurdly large "raster" upload is rejected
    with pytest.raises(MultimodalError):
        guard_image(content_type="image/tiff", size=MAX_RASTER_BYTES + 1)

    # non-raster images are unaffected by the raised ceiling
    with pytest.raises(MultimodalError):
        guard_image(content_type="image/png", size=MAX_IMAGE_BYTES + 1)
