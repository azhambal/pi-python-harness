"""The exact Photon 0.3.4 WASM shipped by Pi, hosted directly in Python.

Only the byte/pixel API is exposed; browser DOM imports are deliberately unavailable,
just as in Node. Each operation owns a Store, so a decoder panic cannot poison the
next call. The compiled module is cached. No JavaScript or Node runtime is needed.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Any

import wasmtime


@cache
def _module() -> tuple[wasmtime.Engine, wasmtime.Module]:
    engine = wasmtime.Engine()
    data = files("pi_agent").joinpath("data/wasm/photon-0.3.4.wasm").read_bytes()
    return engine, wasmtime.Module(engine, data)


class Photon:
    def __init__(self) -> None:
        engine, module = _module()
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        for imported in module.imports:
            assert isinstance(imported.type, wasmtime.FuncType)
            assert imported.name is not None
            callback = (
                self._initialize_refs
                if imported.name == "__wbindgen_init_externref_table"
                else self._unavailable
            )
            linker.define_func(imported.module, imported.name, imported.type, callback, access_caller=True)
        self.exports = linker.instantiate(self.store, module).exports(self.store)
        self.call("__wbindgen_start")

    def __enter__(self) -> Photon:
        return self

    def __exit__(self, *args: Any) -> None:
        self.store.close()

    @staticmethod
    def _unavailable(caller: wasmtime.Caller, *args: Any) -> Any:
        raise wasmtime.Trap("Photon browser API or decoder panic")

    def _initialize_refs(self, caller: wasmtime.Caller) -> None:
        table = self.exports["__wbindgen_export_2"]
        assert isinstance(table, wasmtime.Table)
        offset = table.grow(self.store, 4, wasmtime.Val.externref(None))
        for index, value in enumerate((None, None, True, False)):
            table.set(self.store, offset + index, wasmtime.Val.externref(value))
        table.set(self.store, 0, wasmtime.Val.externref(None))

    def call(self, name: str, *args: Any) -> Any:
        function = self.exports[name]
        assert isinstance(function, wasmtime.Func)
        return function(self.store, *args)

    @property
    def memory(self) -> wasmtime.Memory:
        memory = self.exports["memory"]
        assert isinstance(memory, wasmtime.Memory)
        return memory

    def put(self, data: bytes) -> int:
        pointer = self.call("__wbindgen_malloc", len(data), 1)
        self.memory.write(self.store, data, pointer)
        return pointer

    def load(self, data: bytes) -> int:
        return self.call("photonimage_new_from_byteslice", self.put(data), len(data))

    def from_pixels(self, pixels: bytes, width: int, height: int) -> int:
        return self.call("photonimage_new", self.put(pixels), len(pixels), width, height)

    def take_bytes(self, name: str, *args: int) -> bytes:
        pointer, length = self.call(name, *args)
        data = bytes(self.memory.read(self.store, pointer, pointer + length))
        self.call("__wbindgen_free", pointer, length, 1)
        return data

    def free(self, image: int) -> None:
        self.call("__wbg_photonimage_free", image, 0)

    def dimensions(self, image: int) -> tuple[int, int]:
        return self.call("photonimage_get_width", image), self.call("photonimage_get_height", image)

    def encode(self, image: int, quality: int | None = None) -> bytes:
        if quality is None:
            return self.take_bytes("photonimage_get_bytes", image)
        return self.take_bytes("photonimage_get_bytes_jpeg", image, quality)
