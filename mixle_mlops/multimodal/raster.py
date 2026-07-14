"""Large-raster (GeoTIFF and friends) support: turn an arbitrarily large georeferenced raster into a set of
small, model-sized PNG tiles, each carrying the ground extent (and pixel scale) it covers.

A single 6-band 100k x 100k GeoTIFF is orders of magnitude too big to ever hand a vision model directly (both
the byte count and the pixel count blow past any adapter's limits). ``tile_raster`` opens the bytes with
``rasterio``, downsamples to a fixed megapixel budget if the source is bigger than that, then windows the
(decimated) array into ``max_tile_px``-sized tiles with a small overlap so features that straddle a tile
boundary aren't lost. Each tile keeps the affine-transform-derived ground ``extent`` and ``scale`` (ground
units per pixel) it covers, so downstream code (D5's georeference sidecar, I1's map digitisation) can place
model output back onto the ground without re-deriving it from the source file.

``rasterio``/``pillow`` are optional, lazily imported ``raster``-extra dependencies (install with
``pip install 'mixle-mlops[raster]'``) — this module degrades with a clear ``ImportError`` if they're absent,
never at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from ..core.adapters import ImagePart
from .content import guard_image
from .store import BlobStore


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "reading large rasters needs the optional dependency 'rasterio'. "
            "Install it with: pip install 'mixle-mlops[raster]'  (or: pip install rasterio)."
        ) from exc
    return rasterio


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "encoding raster tiles needs the optional dependency 'pillow'. "
            "Install it with: pip install 'mixle-mlops[raster]'  (or: pip install pillow)."
        ) from exc
    return Image


@dataclass
class RasterTile:
    """One windowed, PNG-encoded slice of a (possibly decimated) raster."""

    png: bytes
    row: int
    col: int
    extent: tuple[float, float, float, float]  # (minx, miny, maxx, maxy), source CRS ground units
    scale: float  # ground units per pixel this tile was rendered at


def _to_uint8(arr: "Any") -> "Any":
    """Per-band min/max stretch to ``uint8`` so any source dtype (float, uint16, ...) becomes a viewable PNG."""
    import numpy as np

    arr = np.asarray(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for i in range(arr.shape[0]):
        band = arr[i].astype(np.float64)
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            out[i] = 0
        else:
            stretched = (band - lo) / (hi - lo) * 255.0
            out[i] = np.clip(stretched, 0, 255).astype(np.uint8)
    return out


def _encode_png(tile_arr: "Any", image_module: "Any") -> bytes:
    """Encode a ``(bands, h, w)`` uint8 array to PNG bytes (grayscale for 1 band, RGB for 3+)."""
    import numpy as np

    bands = tile_arr.shape[0]
    if bands == 1:
        img = image_module.fromarray(tile_arr[0], mode="L")
    else:
        hwc = np.ascontiguousarray(np.transpose(tile_arr[:3], (1, 2, 0)))
        img = image_module.fromarray(hwc, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tile_raster(
    data: bytes,
    *,
    max_tile_px: int = 1024,
    overlap_px: int = 64,
    downsample_to_mp: float | None = 24.0,
) -> list[RasterTile]:
    """Open a raster from bytes and slice it into PNG tiles, each with a ground ``extent`` + ``scale``.

    If the source is bigger than ``downsample_to_mp`` megapixels, it's read at a decimated resolution first
    (never the full-resolution array is materialized) — a factor ``ceil(sqrt(w*h / (downsample_to_mp*1e6)))``
    computed from the source dimensions. The (decimated) array is then windowed into ``max_tile_px``-square
    tiles, each padded by ``overlap_px`` on every interior edge so a feature straddling a tile boundary is
    still fully visible in at least one tile.
    """
    _require_rasterio()
    image_module = _require_pillow()
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile

    with MemoryFile(data) as mem, mem.open() as src:
        width, height = src.width, src.height
        transform = src.transform
        band_count = src.count

        factor = 1.0
        if downsample_to_mp is not None and downsample_to_mp > 0:
            budget = downsample_to_mp * 1e6
            if width * height > budget:
                factor = math.ceil(math.sqrt((width * height) / budget))

        out_width = max(1, int(width / factor))
        out_height = max(1, int(height / factor))

        indexes = list(range(1, min(band_count, 3) + 1))
        arr = src.read(
            indexes,
            out_shape=(len(indexes), out_height, out_width),
            resampling=Resampling.average,
        )
        # Affine for the decimated array: derived from the *actual* out_width/out_height (not the raw
        # factor), so pixel (out_width, out_height) always maps exactly onto the source's far corner
        # regardless of integer-division rounding.
        decim_transform = transform * transform.scale(width / out_width, height / out_height)

    arr = _to_uint8(arr)

    tiles: list[RasterTile] = []
    stride = max_tile_px
    for row0 in range(0, out_height, stride):
        for col0 in range(0, out_width, stride):
            r0 = max(0, row0 - overlap_px)
            c0 = max(0, col0 - overlap_px)
            r1 = min(out_height, row0 + stride + overlap_px)
            c1 = min(out_width, col0 + stride + overlap_px)
            tile_arr = arr[:, r0:r1, c0:c1]
            if tile_arr.size == 0:
                continue

            png_bytes = _encode_png(tile_arr, image_module)
            guard_image(content_type="image/png", size=len(png_bytes))

            x0, y0 = decim_transform * (c0, r0)
            x1, y1 = decim_transform * (c1, r1)
            extent = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            scale = abs(decim_transform.a)

            tiles.append(
                RasterTile(
                    png=png_bytes,
                    row=row0 // stride,
                    col=col0 // stride,
                    extent=extent,
                    scale=scale,
                )
            )
    return tiles


def raster_to_image_parts(data: bytes, store: BlobStore, **kw: Any) -> list[ImagePart]:
    """Tile a raster and persist each tile in ``store``, returning ``ImagePart``s that reference the blobs
    (not inline data — a raster can produce many tiles, and blob refs keep the message payload small)."""
    tiles = tile_raster(data, **kw)
    parts: list[ImagePart] = []
    for tile in tiles:
        record = store.put(
            tile.png,
            filename=f"raster-tile-r{tile.row}-c{tile.col}.png",
            content_type="image/png",
        )
        parts.append(ImagePart(image_url={"file_id": record.id, "url": record.url}))
    return parts
