from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional


def _torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def cuda_enabled(device: str) -> bool:
    torch = _torch()
    return bool(torch is not None and str(device).lower() == "cuda" and torch.cuda.is_available())


def reset_cuda_peak_memory(device: str) -> None:
    torch = _torch()
    if torch is None or not cuda_enabled(device):
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def capture_cuda_peak_memory(device: str) -> Dict[str, Any]:
    torch = _torch()
    if torch is None or not cuda_enabled(device):
        return {
            "device": str(device),
            "cuda_enabled": False,
            "peak_vram_reserved_mb": None,
            "peak_vram_allocated_mb": None,
        }
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        reserved = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
    except Exception:
        reserved = None
    try:
        allocated = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        allocated = None
    return {
        "device": str(device),
        "cuda_enabled": True,
        "peak_vram_reserved_mb": reserved,
        "peak_vram_allocated_mb": allocated,
    }


def summarize_numeric(values: Iterable[float]) -> Dict[str, Optional[float]]:
    seq = [float(v) for v in values]
    if not seq:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(seq)
    idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[idx]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def collect_numeric(records: Iterable[Dict[str, Any]], field: str) -> List[float]:
    out: List[float] = []
    for record in records:
        try:
            value = record.get(field)
            if value is None:
                continue
            out.append(float(value))
        except Exception:
            continue
    return out
