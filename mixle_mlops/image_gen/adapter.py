"""``ImageGenAdapter`` — an image-generation :class:`ModelAdapter`.

Two backends share one adapter:

* ``openai`` (default when a base URL is configured): POST ``{model, prompt, n, size, response_format:"b64_json"}``
  to an OpenAI-compatible ``{base_url}/images/generations`` server (DALL·E-style / a local SD server / a hosted
  API). The returned base64 images are decoded and stored in the platform :class:`BlobStore`.
* local synthetic mode: dependency-free. Synthesizes a tiny PNG per requested
  image so the platform runs end-to-end without an external image service.

Every generated image is written to the blob store and returned as ``{id, url, b64_json}`` — ``url`` being the
gateway path (``/v1/files/{id}/content``) that serves the bytes back, mirroring the multimodal upload flow.

Being a ``ModelAdapter`` (``kind='image'``) it lives in the same registry as LLM/mixle models and shows up in
``/v1/models``; ``chat()`` politely refuses since it is not a chat model.
"""
from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, AsyncIterator

from ..core.adapters import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatRequest,
    ModelAdapter,
    ModelInfo,
)
from ..multimodal.store import BlobStore, get_blob_store

# Standard OpenAI image sizes. Local synthetic mode records the requested size
# while keeping the generated PNG tiny.
_DEFAULT_SIZE = "1024x1024"


def _png_1x1(rgba: tuple[int, int, int, int] = (127, 127, 127, 255)) -> bytes:
    """Return a valid 1x1 RGBA PNG with the given pixel colour.

    Hand-built (no Pillow dependency): IHDR + a single zlib-compressed scanline + IEND."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)  # 1x1, 8-bit, RGBA
    raw = bytes([0]) + bytes(rgba)                        # filter byte 0 + one RGBA pixel
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class ImageGenAdapter(ModelAdapter):
    """Adapter for an OpenAI-compatible image-generation backend or local synthetic mode."""

    kind = "image"

    def __init__(
        self,
        name: str,
        *,
        backend: str = "stub",
        base_url: str = "",
        api_key: str = "",
        upstream_model: str | None = None,
        timeout: float = 600.0,
        store: BlobStore | None = None,
    ):
        self._name = name
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.upstream_model = upstream_model or name
        self.timeout = timeout
        self._store = store

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> set[str]:
        return {"image_generation"}

    def info(self) -> ModelInfo:
        # ``ModelAdapter.kind`` is "image", but the shared ``ModelInfo.kind`` Literal only admits
        # llm/mixle/composite; report "composite" there and advertise the real capability via capabilities().
        return ModelInfo(id=self.name, kind="composite", capabilities=sorted(self.capabilities()))

    @property
    def store(self) -> BlobStore:
        return self._store if self._store is not None else get_blob_store()

    # --- chat: refuse, this is an image model ---
    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        # An image model is not a chat model; surface a clear note rather than pretend to chat.
        raise NotImplementedError(
            f"model {self._name!r} is an image-generation model; use /v1/images/generations, not chat"
        )
        yield  # pragma: no cover - makes this an async generator

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        from ..core.adapters import ChatChoice, ChatMessage

        note = (f"[{self._name}] is an image-generation model. Call POST /v1/images/generations "
                f"with {{model, prompt, n, size}} instead of the chat endpoint.")
        return ChatCompletion(
            model=req.model or self._name,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=note),
                                finish_reason="stop")],
        )

    # --- the actual capability ---
    async def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        size: str | None = None,
        **opts: Any,
    ) -> list[dict[str, Any]]:
        """Generate ``n`` images for ``prompt``, store each in the blob store, return ``[{id, url, b64_json}]``."""
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt must be a non-empty string")
        n = max(1, int(n))
        size = size or _DEFAULT_SIZE

        if self.backend in ("stub", "echo") or not self.base_url:
            images_b64 = self._stub_images(prompt, n)
        else:
            images_b64 = await self._openai_images(prompt, n, size, opts)

        out: list[dict[str, Any]] = []
        store = self.store
        for b64 in images_b64:
            data = base64.b64decode(b64)
            record = store.put(data, filename=f"{self._name}.png", content_type="image/png")
            out.append({"id": record.id, "url": record.url, "b64_json": b64})
        return out

    def _stub_images(self, prompt: str, n: int) -> list[str]:
        # Vary the grey level by index so multiple local synthetic images differ.
        images: list[str] = []
        for i in range(n):
            level = (96 + (i * 32 + (hash(prompt) & 0x3F)) % 128) & 0xFF
            png = _png_1x1((level, level, level, 255))
            images.append(base64.b64encode(png).decode("ascii"))
        return images

    async def _openai_images(
        self, prompt: str, n: int, size: str, opts: dict[str, Any]
    ) -> list[str]:
        import httpx  # lazy: only needed for a real backend

        body: dict[str, Any] = {
            "model": self.upstream_model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }
        body.update({k: v for k, v in opts.items() if v is not None})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/images/generations", json=body, headers=headers)
            r.raise_for_status()
            data = r.json()

        images: list[str] = []
        for item in data.get("data", []):
            if item.get("b64_json"):
                images.append(item["b64_json"])
            elif item.get("url"):
                # Backend returned a URL instead of inline bytes: fetch and re-encode so we always own the bytes.
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    img = await client.get(item["url"])
                    img.raise_for_status()
                    images.append(base64.b64encode(img.content).decode("ascii"))
        if not images:
            raise RuntimeError("image backend returned no images")
        return images


def register_demo_image_model(registry, name: str = "stub-image", store: BlobStore | None = None):
    """Register a dependency-free image model for local ``/v1/images/generations`` smoke tests."""
    return registry.register(ImageGenAdapter(name, backend="stub", store=store))
