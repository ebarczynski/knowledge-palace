"""Embedding service: text to vectors with the fastest backend that works.

Backend order for ``provider = "auto"``:

1. **MLX** — only when ``mlx-embeddings`` is installed *and* it ships a module
   for the configured model's architecture. The default model
   (``nomic-embed-text-v1.5``, architecture ``nomic_bert``) is not supported
   upstream, so MLX is skipped for it.
2. **ONNX Runtime** — the usual choice: ``optimum`` + ``onnxruntime`` run the
   model's exported ``onnx/model.onnx`` graph on ``CPUExecutionProvider``.
3. **PyTorch** — fallback when ONNX cannot load.

The ONNX and PyTorch backends produce numerically equivalent vectors (they were
compared at differences around 1e-7), so which one ran never changes what is
stored. Their speed is a property of the machine: measure on yours rather than
trusting a number from a README.
"""

from __future__ import annotations

import asyncio
import json
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from ..config import EmbeddingConfig

# Status/loading messages go to stderr so they never corrupt protocols that
# use stdout for data (e.g. the MCP stdio JSON-RPC stream). They still appear
# in the terminal for CLI/serve use.
console = Console(stderr=True)

# Nomic embed models are instruction-tuned: queries and documents MUST carry
# task prefixes or retrieval quality degrades sharply (all texts land in the
# same "document" mode, compressing the similarity space so unrelated queries
# score ~0.65+ against everything). Other model families must NOT be prefixed.
_TASK_PREFIX_MODELS = ("nomic-embed", "nomic-embed-text")
_QUERY_PREFIX = "search_query: "
_DOC_PREFIX = "search_document: "


def _wants_task_prefixes(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(marker in lowered for marker in _TASK_PREFIX_MODELS)


class EmbeddingService:
    """Generates embeddings using the fastest available backend."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._backend: str | None = None  # "mlx", "onnx", "pytorch"
        self._use_prefixes = _wants_task_prefixes(config.model)
        self._loading: asyncio.Lock | None = None

    @property
    def backend(self) -> str | None:
        """Backend in use once the model is loaded: ``mlx``, ``onnx`` or ``pytorch``."""
        return self._backend

    async def load(self) -> None:
        """Load the model now (in a thread pool); concurrent callers share one load."""
        if self._model is not None:
            return
        if self._loading is None:
            self._loading = asyncio.Lock()
        async with self._loading:
            if self._model is None:
                await asyncio.get_running_loop().run_in_executor(None, self._load_model)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _supported_mlx_architectures(self) -> set[str]:
        """Return the model architectures `mlx_embeddings` can load."""
        try:
            import mlx_embeddings.models as m
        except ImportError:
            return set()
        # iter_modules() yields (finder, name, ispkg) tuples; the architecture
        # is the module *name*.
        return {name for _, name, _ in pkgutil.iter_modules(m.__path__)} - {"base"}

    def _is_mlx_supported(self) -> bool:
        """True if mlx-embeddings can load the configured model's architecture.

        We read the model's `model_type` from its HuggingFace config and check
        it against the architectures mlx-embeddings ships modules for. This
        avoids a guaranteed-to-fail model load (and its noisy traceback) for
        unsupported architectures such as nomic_bert. Without mlx-embeddings
        installed there is nothing to probe, so no config is ever fetched.
        """
        supported = self._supported_mlx_architectures()
        if not supported:
            return False
        try:
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(self.config.model, "config.json")
            model_type = json.loads(Path(cfg_path).read_text(encoding="utf-8")).get("model_type", "")
        except Exception:  # noqa: BLE001 - a best-effort probe; the real load surfaces any error
            # Can't read config — let _load_mlx attempt and surface the real error.
            return True
        # mlx_embeddings maps model_type -> submodule name (dashes -> underscores)
        return model_type.replace("-", "_") in supported

    def _load_mlx(self):
        """Try to load model with MLX backend (Apple Silicon only)."""
        import mlx.core as mx
        from mlx_embeddings.utils import generate as mlx_generate
        from mlx_embeddings.utils import load as mlx_load

        console.print(f"Loading embedding model: [cyan]{self.config.model}[/cyan] [dim](MLX)[/dim]")

        model, tokenizer = mlx_load(self.config.model)
        return model, tokenizer, mx, mlx_generate

    def _local_model_path(self) -> str:
        """A directory holding the model, resolved from the local cache.

        sentence-transformers calls the Hugging Face Hub whenever it is given a
        repo id, even when every file is already cached, and fails outright
        offline. Handing it the snapshot directory instead makes loading work
        without network access. A model that is already a local directory is
        used as is.
        """
        path = Path(self.config.model).expanduser()
        if path.is_dir():
            return str(path)
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        # This runs on every model load; a "Fetching 20 files" bar for files that
        # are already cached is noise, and stderr is where MCP clients log.
        disable_progress_bars()
        return snapshot_download(self.config.model)

    def _load_onnx(self):
        """Try to load model with ONNX Runtime backend."""
        from sentence_transformers import SentenceTransformer

        console.print(f"Loading embedding model: [cyan]{self.config.model}[/cyan] [dim](ONNX)[/dim]")
        return SentenceTransformer(
            self._local_model_path(),
            device=self.config.device,
            # nomic-embed-text-v1.5 has no native config class in transformers:
            # loading it with trust_remote_code=False raises a ValueError. This
            # executes the model author's code from their nomic-ai repos, at
            # whatever revision they publish -- a trust decision, not a detail.
            trust_remote_code=True,
            backend="onnx",
            model_kwargs={"provider": "CPUExecutionProvider", "file_name": "onnx/model.onnx"},
        )

    def _load_pytorch(self):
        """Load model with PyTorch backend (fallback)."""
        from sentence_transformers import SentenceTransformer

        console.print(f"Loading embedding model: [cyan]{self.config.model}[/cyan] [dim](PyTorch)[/dim]")
        return SentenceTransformer(
            self._local_model_path(),
            device=self.config.device,
            trust_remote_code=True,  # see _load_onnx
        )

    def _load_model(self):
        """Load the model with the configured backend.

        ``provider`` is honored: "onnx"/"pytorch" force that backend (useful
        when e.g. ONNX's CPU thread pool eats RAM on long sequences); "auto"
        keeps the MLX → ONNX → PyTorch preference ladder.
        """
        if self._model is not None:
            return self._model

        if self.config.provider == "pytorch":
            self._model = self._load_pytorch()
            self._backend = "pytorch"
            console.print("[green]Model loaded (PyTorch backend - forced by provider)[/green]")
            return self._model

        if self.config.provider == "onnx":
            self._model = self._load_onnx()
            self._backend = "onnx"
            console.print("[green]Model loaded (ONNX backend - forced by provider)[/green]")
            return self._model

        # MLX: only attempt when the model architecture is actually supported
        # by mlx-embeddings (e.g. nomic_bert is not — see _is_mlx_supported).
        if self._is_mlx_supported():
            try:
                self._model = self._load_mlx()
                self._backend = "mlx"
                console.print("[green]Model loaded (MLX backend - Apple Metal)[/green]")
                return self._model
            except Exception as e:
                console.print(f"[dim]MLX unavailable: {e}[/dim]")
        else:
            console.print(
                f"[dim]MLX skipped: architecture of "
                f"{self.config.model} is not supported by mlx-embeddings[/dim]"
            )

        try:
            self._model = self._load_onnx()
            self._backend = "onnx"
            console.print("[green]Model loaded (ONNX Runtime backend)[/green]")
            return self._model
        except Exception as e:
            console.print(f"[yellow]ONNX unavailable ({type(e).__name__}: {e}); using PyTorch instead[/yellow]")

        self._model = self._load_pytorch()
        self._backend = "pytorch"
        console.print("[green]Model loaded (PyTorch backend)[/green]")
        return self._model

    async def embed(self, text: str) -> list[float] | None:
        """Generate embedding for a single text."""
        if not text.strip():
            return None
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embeddings for a batch of *documents* (indexed content).

        For nomic models the ``search_document: `` task prefix is applied per
        text — required for the model's retrieval mode to work correctly.
        """
        if not texts:
            return []

        model_bundle = self._load_model()

        # Filter empty texts
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [None] * len(texts)

        if self._use_prefixes:
            non_empty_texts = [_DOC_PREFIX + t for t in non_empty_texts]

        # Run embedding in thread pool (CPU/Metal-bound)
        loop = asyncio.get_running_loop()

        if self._backend == "mlx":
            embeddings = await loop.run_in_executor(
                None, self._encode_mlx, model_bundle, non_empty_texts
            )
        else:
            # ONNX or PyTorch (sentence-transformers)
            embeddings = await loop.run_in_executor(
                None, self._encode_sentence_transformers, model_bundle, non_empty_texts
            )

        # Convert to float16 lists for storage (halves DB storage)
        result: list[list[float] | None] = [None] * len(texts)
        for idx, embedding in zip(non_empty_indices, embeddings, strict=True):
            result[idx] = embedding.astype(np.float16).tolist()

        return result

    def _encode_mlx(self, model_bundle, texts: list[str]) -> np.ndarray:
        """Encode texts using MLX backend."""
        model, tokenizer, mx, mlx_generate = model_bundle

        all_embeddings = []
        batch_size = self.config.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            output = mlx_generate(model, tokenizer, texts=batch)
            # output.text_embeds is an mx.array (normalized)
            embeds = np.array(output.text_embeds)
            all_embeddings.append(embeds)
            # Keep Metal pipeline moving
            mx.eval(all_embeddings[-1])

        if len(all_embeddings) > 1:
            return np.concatenate(all_embeddings, axis=0)
        return all_embeddings[0]

    def _encode_sentence_transformers(self, model, texts: list[str]) -> np.ndarray:
        """Encode texts using sentence-transformers (ONNX or PyTorch backend)."""
        return model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query.

        Applies the ``search_query: `` task prefix for nomic models so the
        query lands in the retrieval mode matching the prefixed documents.

        Raises:
            ValueError: on an empty or whitespace-only query.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty query")
        model_bundle = self._load_model()

        prefixed = (_QUERY_PREFIX + text) if self._use_prefixes else text
        loop = asyncio.get_running_loop()
        if self._backend == "mlx":
            embeddings = await loop.run_in_executor(
                None, self._encode_mlx, model_bundle, [prefixed]
            )
        else:
            embeddings = await loop.run_in_executor(
                None, self._encode_sentence_transformers, model_bundle, [prefixed]
            )
        return embeddings[0].astype(np.float16).tolist()


__all__ = ["EmbeddingService"]
