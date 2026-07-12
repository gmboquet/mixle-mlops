"""Image generation: an OpenAI-compatible ``/v1/images/generations`` backend wrapped as a ``ModelAdapter``.

The :class:`~mixle_mlops.image_gen.adapter.ImageGenAdapter` POSTs to an OpenAI-compatible images backend (config
``MIXLE_IMAGE_BASE_URL`` / ``MIXLE_IMAGE_API_KEY`` / ``MIXLE_IMAGE_MODEL``) and stashes the returned image bytes in
the platform :class:`BlobStore`, returning ``{id, url}`` per image. A dependency-free local mode emits a tiny PNG so
the feature runs end-to-end without an external image service."""
from .adapter import ImageGenAdapter, register_demo_image_model

__all__ = ["ImageGenAdapter", "register_demo_image_model"]
