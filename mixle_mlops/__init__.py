"""mixle-mlops: an all-in-one AI platform.

Host mixle's probabilistic models *and* open LLMs (Llama, DeepSeek, ...) behind one OpenAI-compatible gateway,
let mixle compose them, and ship the full product surface — accounts, API keys, multimodal I/O, an MCP server,
a chat UI + landing page, and a principled (mixle-powered) RLHF feedback loop. See the Sphinx manual's
overview and package map (``docs/overview.rst``, ``docs/package-map.rst``) for the package boundaries.

    from mixle_mlops.gateway import create_app
    app = create_app()              # a FastAPI app; run with `mixle-serve`
"""

__version__ = "0.7.0"
