from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024**3)


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def module_parameter_bytes(module: Any) -> int:
    if module is None or not hasattr(module, "parameters"):
        return 0
    return sum(tensor_nbytes(param) for param in module.parameters())


def cuda_memory_snapshot(device: Any = None) -> dict[str, Any]:
    try:
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        if device is not None:
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
        else:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
        return {
            "cuda_available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": int(free),
            "total_bytes": int(total),
            "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated(device)),
            "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved(device)),
            "max_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated(device)),
            "max_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved(device)),
            "free_gib": bytes_to_gib(free),
            "total_gib": bytes_to_gib(total),
        }
    except Exception as exc:
        return {"cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def _maybe_gib_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    for key, value in values.items():
        if key.endswith("_bytes") and isinstance(value, int):
            result[key[: -len("_bytes")] + "_gib"] = bytes_to_gib(value)
    return result


@dataclass
class MemoryTrace:
    enabled: bool = False
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        stage: str,
        *,
        device: Any = None,
        estimates: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        entry: dict[str, Any] = {
            "stage": stage,
            "cuda": cuda_memory_snapshot(device),
        }
        if estimates is not None:
            entry["estimates"] = _maybe_gib_fields(estimates)
        if extra:
            entry.update(dict(extra))
        self.stages.append(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                **entry,
                "cuda": dict(entry.get("cuda", {})),
                "estimates": dict(entry.get("estimates", {})),
            }
            for entry in self.stages
        ]
