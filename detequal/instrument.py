from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


def tensor_hash(t: torch.Tensor) -> str:
    with torch.no_grad():
        cpu = t.detach().to("cpu").contiguous()
        header = f"{cpu.dtype}|{tuple(cpu.shape)}|".encode()
        return hashlib.sha1(header + cpu.numpy().tobytes()).hexdigest()


@dataclass
class Snapshot:
    canonical_name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    hash: str
    tensor: torch.Tensor | None = None


@dataclass
class Capture:
    run_id: int
    full_tensors: bool = False
    tensors: dict[str, Snapshot] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)
    _call_index: dict[str, int] = field(default_factory=dict)

    def _canonical(self, module_path: str, op_type: str) -> str:
        idx = self._call_index.get(module_path, 0)
        self._call_index[module_path] = idx + 1
        return f"{module_path}::{op_type}::call{idx}"

    def _record(self, module_path: str, op_type: str, kind: str, t: torch.Tensor) -> None:
        if not torch.is_tensor(t):
            return
        name = self._canonical(module_path, kind)
        self.tensors[name] = Snapshot(
            canonical_name=name,
            kind=kind,
            dtype=str(t.dtype),
            shape=tuple(t.shape),
            hash=tensor_hash(t),
            tensor=(t.detach().to("cpu").clone() if self.full_tensors else None),
        )

    def attach(self, model: torch.nn.Module) -> None:
        for module_path, module in model.named_modules():
            if module is model and module_path == "":
                continue

            type_name = type(module).__name__

            def fwd_hook(mod, inputs, output, _p=module_path, _t=type_name):
                out = output[0] if isinstance(output, (tuple, list)) and output else output
                if torch.is_tensor(out):
                    self._record(_p, _t, "forward", out)

            self._handles.append(module.register_forward_hook(fwd_hook))

            if hasattr(module, "register_full_backward_hook"):

                def bwd_hook(mod, grad_input, grad_output, _p=module_path, _t=type_name):
                    g = grad_output[0] if grad_output else None
                    if torch.is_tensor(g):
                        self._record(_p, _t, "backward", g)

                self._handles.append(module.register_full_backward_hook(bwd_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


_ACTIVE_MODEL: torch.nn.Module | None = None


def bind_model(model: torch.nn.Module) -> None:
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model


@contextmanager
def capture(run_id: int, *, full_tensors: bool = False) -> Iterator[Capture]:
    cap = Capture(run_id=run_id, full_tensors=full_tensors)
    if _ACTIVE_MODEL is not None:
        cap.attach(_ACTIVE_MODEL)
    try:
        yield cap
    finally:
        cap.detach()


class DispatchCapture:
    def __enter__(self):
        raise NotImplementedError(
            "dispatch-level capture is not implemented yet; use module hooks via capture()"
        )

    def __exit__(self, *exc):
        return False
