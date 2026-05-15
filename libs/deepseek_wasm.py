# deepseek_wasm.py
import pathlib
import struct
from typing import Any, Optional

from wasmtime import Store, Module, Instance, Func, Memory


class DeepSeekHashWasm:
    def __init__(self, wasm_path: str | None = None) -> None:
        if wasm_path is None:
            wasm_path = "sha3_wasm_bg.7b9ca65ddd.wasm"

        wasm_file = pathlib.Path(wasm_path)
        if not wasm_file.exists():
            raise FileNotFoundError(f"WASM file not found: {wasm_file}")

        self.store = Store()
        self.module = Module.from_file(self.store.engine, str(wasm_file))

        imports: list[Any] = []
        self.instance = Instance(self.store, self.module, imports)

        self.exports = self.instance.exports(self.store)

        self._add_to_stack_pointer: Func = self.exports["__wbindgen_add_to_stack_pointer"]
        self._malloc: Func = self.exports["__wbindgen_export_0"]
        self._realloc: Func = self.exports["__wbindgen_export_1"]
        self._wasm_solve: Func = self.exports["wasm_solve"]
        self.memory: Memory = self.exports["memory"]

    # -------- utilidades de memoria --------
    
    def _write_bytes(self, ptr: int, data: bytes) -> None:
        """Escribe bytes en la memoria lineal."""
        self.memory.write(self.store, data, start=ptr)

    def _read_bytes(self, ptr: int, size: int) -> bytes:
        """Lee bytes de la memoria lineal."""
        return bytes(self.memory.read(self.store, start=ptr, stop=ptr + size))

    def _encode_string(self, text: str) -> tuple[int, int]:
        """Codifica un string a UTF-8 y lo escribe en memoria."""
        encoded = text.encode('utf-8')
        size = len(encoded)
        
        # malloc(size, align=1)
        ptr_val = self._malloc(self.store, size, 1)
        ptr = int(ptr_val)
        
        self._write_bytes(ptr, encoded)
        return ptr, size

    # -------- API principal --------

    def calculate_hash(
        self,
        algorithm: str,
        challenge: str,
        salt: str,
        difficulty: int,
        expire_at: int,
    ) -> Optional[int]:
        """
        Resuelve el PoW y retorna el nonce encontrado (answer).
        Retorna None si no encuentra solución.
        """
        if algorithm != "DeepSeekHashV1":
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        prefix = f"{salt}_{expire_at}_"

        # Ajustar stack pointer
        retptr_val = self._add_to_stack_pointer(self.store, -16)
        retptr = int(retptr_val)

        try:
            # Codificar strings
            ptr0, len0 = self._encode_string(challenge)
            ptr1, len1 = self._encode_string(prefix)

            # Llamar a la función WASM con difficulty como float
            self._wasm_solve(
                self.store,
                retptr,      # i32
                ptr0,        # i32
                len0,        # i32
                ptr1,        # i32
                len1,        # i32
                float(difficulty),  # f64
            )

            # Leer resultados
            status_bytes = self._read_bytes(retptr, 4)
            value_bytes = self._read_bytes(retptr + 8, 8)
            
            status = struct.unpack("<i", status_bytes)[0]
            value = struct.unpack("<d", value_bytes)[0]

            # status == 1 significa que encontró solución
            # value es el nonce (como float, pero debería ser entero)
            if status == 0:
                return None
            
            # Convertir a entero (el nonce debe ser int64)
            return int(value)

        finally:
            # Restaurar stack pointer
            self._add_to_stack_pointer(self.store, 16)