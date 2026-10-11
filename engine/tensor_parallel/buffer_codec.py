"""
Zero-copy and low-overhead tensor buffer serialization for gRPC collective operations.
Optimized for commodity CPU/network transmission avoiding python object serialization bottlenecks.
"""

from typing import Tuple
import numpy as np

# PyTorch will be imported dynamically or globally
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Mapping between string dtypes and numpy / torch dtypes
DTYPE_MAP_NP = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
}


def serialize_tensor(tensor) -> Tuple[bytes, Tuple[int, ...], str]:
    """
    Serializes a PyTorch tensor or NumPy array into raw bytes, shape, and dtype string.
    Zero-overhead binary buffer extraction.
    """
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        t_cpu = tensor.detach().contiguous().cpu()
        # Handle bfloat16 if needed
        if t_cpu.dtype == torch.bfloat16:
            # View as int16 bytes or convert to float16/float32
            # For simplicity and broad compatibility on commodity CPU:
            np_arr = t_cpu.view(torch.int16).numpy()
            return np_arr.tobytes(), tuple(tensor.shape), "bfloat16"
        else:
            np_arr = t_cpu.numpy()
            dtype_str = str(np_arr.dtype)
            return np_arr.tobytes(), tuple(tensor.shape), dtype_str
    elif isinstance(tensor, np.ndarray):
        np_arr = np.ascontiguousarray(tensor)
        return np_arr.tobytes(), tuple(tensor.shape), str(np_arr.dtype)
    else:
        raise TypeError(f"Unsupported tensor type for serialization: {type(tensor)}")


def deserialize_tensor(raw_bytes: bytes, shape: Tuple[int, ...], dtype_str: str, device: str = "cpu"):
    """
    Reconstructs a PyTorch tensor (or NumPy array) from raw contiguous bytes.
    """
    if dtype_str == "bfloat16":
        np_arr = np.frombuffer(raw_bytes, dtype=np.int16).reshape(shape)
        if HAS_TORCH:
            t = torch.from_numpy(np_arr).view(torch.bfloat16)
            return t.to(device)
        return np_arr

    np_dtype = DTYPE_MAP_NP.get(dtype_str, np.float32)
    np_arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

    if HAS_TORCH:
        # np_arr from buffer is read-only in some cases, copy if needed for PyTorch mutable tensor
        t = torch.from_numpy(np_arr.copy())
        return t.to(device)
    return np_arr
