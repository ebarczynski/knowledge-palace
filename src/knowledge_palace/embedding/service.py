"""Embedding service using sentence-transformers with ONNX acceleration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from ..config import EmbeddingConfig

console = Console()


class EmbeddingService:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        """Lazy-load the model with ONNX backend if available."""
        if self._model is None:
            console.print(f"Loading embedding model: [cyan]{self.config.model}[/cyan]")
            from sentence_transformers import SentenceTransformer

            # Try ONNX backend first (2-3x faster, lower memory)
            backend = "openvino"  # try in order of preference
            try:
                self._model = SentenceTransformer(
                    self.config.model,
                    device=self.config.device,
                    trust_remote_code=True,
                    backend="onnx",
                    model_kwargs={
                        "provider": "CPUExecutionProvider",
                    },
                )
                console.print("[green]Model loaded (ONNX backend)[/green]")
            except Exception:
                try:
                    self._model = SentenceTransformer(
                        self.config.model,
                        device=self.config.device,
                        trust_remote_code=True,
                        backend="openvino",
                    )
                    console.print("[green]Model loaded (OpenVINO backend)[/green]")
                except Exception:
                    self._model = SentenceTransformer(
                        self.config.model,
                        device=self.config.device,
                        trust_remote_code=True,
                    )
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

        model = self._load_model()

        # Filter empty texts
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [None] * len(texts)

        # Run embedding in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                non_empty_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                # Use float16 to halve memory and speed up computation
                # Precision is fine for similarity search
                convert_to_numpy=True,
            ),
        )

        # Convert to float16 lists for storage (halves DB storage)
        result: list[list[float] | None] = [None] * len(texts)
        for idx, embedding in zip(non_empty_indices, embeddings):
            result[idx] = embedding.astype(np.float16).tolist()

        return result

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query (guaranteed non-None)."""
        result = await self.embed(text)
        if result is None:
            raise ValueError("Cannot embed empty query")
        return result
