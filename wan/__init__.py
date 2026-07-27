"""Wan public API with lazy pipeline imports.

Keeping these imports lazy lets lightweight utilities such as ``wan.analysis``
run without constructing every generation pipeline or importing all optional
runtime dependencies. Public names remain unchanged.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_MODULES = {
    "configs": ".configs",
    "distributed": ".distributed",
    "modules": ".modules",
}
_LAZY_OBJECTS = {
    "WanFLF2V": (".first_last_frame2video", "WanFLF2V"),
    "WanI2V": (".image2video", "WanI2V"),
    "WanT2V": (".text2video", "WanT2V"),
    "WanVace": (".vace", "WanVace"),
    "WanVaceMP": (".vace", "WanVaceMP"),
}

__all__ = [*_LAZY_MODULES, *_LAZY_OBJECTS]


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        value = import_module(_LAZY_MODULES[name], __name__)
    elif name in _LAZY_OBJECTS:
        module_name, object_name = _LAZY_OBJECTS[name]
        value = getattr(import_module(module_name, __name__), object_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
