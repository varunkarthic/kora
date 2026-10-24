"""Adds two vectors on the first KORA device with NumPy.

    python examples/vector_add.py
"""

from pathlib import Path

import numpy as np

import kora

KERNELS = Path(__file__).resolve().parent / "kernels"

runtime = kora.Runtime.create()
metadata = kora.KernelMetadata.from_json((KERNELS / "vector_add.json").read_text())
kernel = runtime.load_kernel(KERNELS / "vector_add.spv", "main", metadata)

count = 1 << 20
a = runtime.create_buffer_from(np.arange(count, dtype=np.uint32))
b = runtime.create_buffer_from(np.full(count, 10, dtype=np.uint32))
sums = runtime.create_buffer(count * 4)

(
    runtime.dispatch(kernel)
    .global_size(count)
    .read(a)
    .read(b)
    .write(sums)
    .push(np.array([count], dtype=np.uint32))
    .submit()
    .wait()
)

result = sums.to_numpy(np.uint32)
mismatches = int(np.count_nonzero(result != np.arange(count, dtype=np.uint32) + 10))
print(f"KORA {kora.version()}: vector add of {count} elements: {mismatches} mismatches")
