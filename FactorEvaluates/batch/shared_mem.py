#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SharedMemory 包装：主进程创建，worker attach。"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Optional

import numpy as np


def _dtype(spec: Any) -> np.dtype:
    return np.dtype(spec)


class ShmArray:
    def __init__(self, shm: SharedMemory, array: np.ndarray, owner: bool):
        self.shm = shm
        self.array = array
        self.owner = owner

    @classmethod
    def create(cls, array: np.ndarray) -> "ShmArray":
        src = np.ascontiguousarray(array)
        shm = SharedMemory(create=True, size=int(src.nbytes))
        view = np.ndarray(src.shape, dtype=src.dtype, buffer=shm.buf)
        view[:] = src
        return cls(shm, view, True)

    @classmethod
    def create_empty(cls, shape, dtype) -> "ShmArray":
        dt = np.dtype(dtype)
        size = int(np.prod(shape) * dt.itemsize)
        shm = SharedMemory(create=True, size=max(size, 1))
        view = np.ndarray(shape, dtype=dt, buffer=shm.buf)
        if np.issubdtype(dt, np.floating):
            view.fill(np.nan)
        else:
            view.fill(0)
        return cls(shm, view, True)

    @classmethod
    def attach(cls, meta: Dict[str, Any]) -> "ShmArray":
        shm = SharedMemory(name=str(meta["name"]))
        view = np.ndarray(tuple(meta["shape"]), dtype=_dtype(meta["dtype"]), buffer=shm.buf)
        return cls(shm, view, False)

    def meta(self) -> Dict[str, Any]:
        return {
            "name": self.shm.name,
            "shape": list(self.array.shape),
            "dtype": str(self.array.dtype),
        }

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        if not self.owner:
            return
        try:
            self.shm.unlink()
        except Exception:
            pass


class ShmPack:
    def __init__(self):
        self.items: Dict[str, ShmArray] = {}

    def add(self, key: str, shm: ShmArray) -> np.ndarray:
        self.items[key] = shm
        return shm.array

    def meta(self) -> Dict[str, Dict[str, Any]]:
        return {key: item.meta() for key, item in self.items.items()}

    def close(self) -> None:
        for item in self.items.values():
            item.close()

    def unlink(self) -> None:
        for item in self.items.values():
            item.unlink()

    def close_and_unlink(self) -> None:
        self.close()
        self.unlink()


def attach_pack(meta: Dict[str, Dict[str, Any]]) -> Dict[str, ShmArray]:
    return {key: ShmArray.attach(spec) for key, spec in meta.items()}


def close_attached(items: Dict[str, ShmArray]) -> None:
    for item in items.values():
        item.close()
