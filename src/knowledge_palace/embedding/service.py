"""Embedding service using MLX on Apple Silicon.

MLX is Apple's machine learning framework optimized for Apple Silicon:
- Uses Metal GPU via unified memory (no CPU-GPU copy overhead)
- 3-5x faster than ONNX Runtime on M-series chips
- Lower memory footprint (shared memory architecture)
- Supports quantization (4-bit/8-bit) for even faster inference

Falls back to ONNX Runtime, then PyTorch if MLX is unavailable.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from ..config import EmbeddingConfig

console = Console()


class EmbeddingService:
    """Generates embeddings using the fastest available backend."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._backend = None  # "mlx", "onnx", "pytorch"

    def _load_mlx(self):
        """Try to load model with MLX backend (Apple Silicon only)."""
        import mlx.core as mx
        from mlx_embeddings.utils import load as mlx_load, generate as mlx_generate

        console.print(f"Loading embedding model: [cyan]{self.config.model}[/cyan] [dim](MLX)[/dim]")

        model, tokenizer = mlx_load(self.config.model)
        return model, tokenizer, mx, mlx_generate

    def _load_onnx(self):
        """Try to load model with ONNX Runtime backend."""
        from sentence_transformers import SentenceTransformer

        console.print(f"Loading embedding model: [cyan]{self.config.model}[/cyan] [dim](ONNX)[/dim]")
        model = SentenceTransformer(
            self.config.model,
            device=self.config.device,
            trust_remote_code=True,
            backend="onnx",
            model_kwargs={"provider": "CPUExecutionProvider"},
        )
        return model

    def _load_pytorch(self):
        """Load model with PyTorch backend (fallback)."""
        from sentence_transformers import SentenceTransformer

        console.print(f"Loading embedding model: [cyan]{self.config.model}[/cyan] [dim](PyTorch)[/dim]")
        model = SentenceTransformer(
            self.config.model,
            device=self.config.device,
            trust_remote_code=True,
        )
        return model

    def _load_model(self):
        """Lazy-load the model with the fastest available backend."""
        if self._model is not None:
            return self._model

        # Try backends in order of preference for Apple Silicon
        try:
            self._model = self._load_mlx()
            self._backend = "mlx"
            console.print("[green]Model loaded (MLX backend - Apple Metal)[/green]")
            return self._model
        except Exception as e:
            console.print(f"[dim]MLX unavailable: {e}[/dim]")

        try:
            self._model = self._load_onnx()
            self._backend = "onnx"
            console.print("[green]Model loaded (ONNX Runtime backend)[/green]")
            return self._model
        except Exception as e:
            console.print(f"[dim]ONNX unavailable: {e}[/dim]")

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
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        model_bundle = self._load_model()

        # Filter empty texts
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [None] * len(texts)

        # Run embedding in thread pool (CPU/Metal-bound)
        loop = asyncio.get_event_loop()

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
        for idx, embedding in zip(non_empty_indices, embeddings):
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
        """Generate embedding for a search query (guaranteed non-None)."""
        result = await self.embed(text)
        if result is None:
            raise ValueError("Cannot embed empty query")
        return result
