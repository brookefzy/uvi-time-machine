"""Shared helpers for the DINOv3 visual-similarity pipeline."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def build_embedding_columns(dim: int) -> list[str]:
    """Return zero-padded embedding column names for a vector dimension."""
    if dim < 0:
        raise ValueError("dim must be non-negative")
    return [f"e_{idx:04d}" for idx in range(dim)]


def discover_embedding_columns(
    df: pd.DataFrame,
    prefix: str = "e_",
    strict: bool = True,
) -> list[str]:
    """Discover embedding columns and sort them by numeric suffix."""
    discovered: list[tuple[int, str]] = []
    seen_suffixes: set[int] = set()

    for column in df.columns:
        if not column.startswith(prefix):
            continue
        suffix = column[len(prefix) :]
        if not suffix.isdigit():
            continue
        suffix_int = int(suffix)
        if suffix_int in seen_suffixes:
            raise ValueError(f"duplicate embedding column suffix: {suffix_int}")
        seen_suffixes.add(suffix_int)
        discovered.append((suffix_int, column))

    discovered.sort(key=lambda item: item[0])
    columns = [column for _, column in discovered]

    if strict and columns:
        expected = set(range(discovered[-1][0] + 1))
        missing = sorted(expected - seen_suffixes)
        if missing:
            preview = ", ".join(f"{prefix}{idx:04d}" for idx in missing[:5])
            raise ValueError(f"missing embedding columns: {preview}")

    if strict and not columns:
        raise ValueError(f"no embedding columns found with prefix {prefix!r}")

    return columns


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row while preserving all-zero rows."""
    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def atomic_write_parquet(df: pd.DataFrame, output_path: Path | str) -> None:
    """Write parquet atomically with the temp file in the destination directory."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.tmp")
    try:
        df.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.replace(output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load an image as RGB and detach it from the open file handle."""
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def resolve_city_file_stem(city: str, override: str | None = None) -> str:
    """Resolve display city names to server file stems."""
    if override:
        return override
    normalized = unicodedata.normalize("NFKD", city)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def _move_inputs_to_device(inputs: dict, device: str) -> dict:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=float)
    return np.asarray(value, dtype=float)


def _extract_model_output(output) -> np.ndarray:
    for attr in ("pooler_output", "last_hidden_state", "image_embeds"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            array = _tensor_to_numpy(value)
            if attr == "last_hidden_state" and array.ndim == 3:
                return array[:, 0, :]
            return array
    if isinstance(output, (tuple, list)) and output:
        return _tensor_to_numpy(output[0])
    return _tensor_to_numpy(output)


def load_embedding_backend(
    model_name: str,
    device: str = "cpu",
    backend: str = "transformers",
    local_files_only: bool = False,
    ignore_mismatched_sizes: bool = False,
):
    """Load a model and processor for an embedding backend."""
    if backend == "transformers":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        try:
            model = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
        except RuntimeError as exc:
            if "ignore_mismatched_sizes" in str(exc):
                raise RuntimeError(
                    "Transformers reported tensor-size mismatches while loading "
                    f"{model_name!r}. If this is an expected mismatch in a staged "
                    "local checkpoint, rerun with --ignore-mismatched-sizes. Use "
                    "that option only after checking the mismatch report; backbone "
                    "tensor mismatches can leave randomly initialized embeddings."
                ) from exc
            raise
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model, processor

    if backend == "timm":
        if ignore_mismatched_sizes:
            raise ValueError("--ignore-mismatched-sizes is only supported by transformers")
        import timm

        model = timm.create_model(model_name, pretrained=not local_files_only, num_classes=0)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model, None

    raise ValueError(f"Unsupported embedding backend: {backend}")


def embed_image_batch(
    paths: Sequence[str | Path],
    model,
    processor,
    device: str,
    image_loader: Callable[[str | Path], object] = load_rgb_image,
) -> np.ndarray:
    """Embed a batch of images with injected model/processor dependencies."""
    images = [image_loader(path) for path in paths]

    if processor is None:
        output = model(images)
    else:
        inputs = processor(images=images, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, device)
        output = model(**inputs)

    return _extract_model_output(output)


def verify_embedding_backend(
    model_name_or_path: str,
    backend: str,
    device: str,
    local_files_only: bool,
    sample_paths: Iterable[str | Path],
    ignore_mismatched_sizes: bool = False,
    backend_loader: Callable[..., tuple] = load_embedding_backend,
    embedder: Callable[..., np.ndarray] = embed_image_batch,
) -> dict:
    """Load an embedding backend and report basic smoke-test diagnostics."""
    paths = list(sample_paths)
    model, processor = backend_loader(
        model_name=model_name_or_path,
        device=device,
        backend=backend,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )
    embeddings = np.asarray(embedder(paths, model, processor, device), dtype=float)
    norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([])
    unit_norm = bool(np.allclose(norms, np.ones_like(norms), atol=1e-5))

    return {
        "model_name": model_name_or_path,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "batch_size": len(paths),
        "device": device,
        "finite": bool(np.isfinite(embeddings).all()),
        "unit_norm": unit_norm,
    }
