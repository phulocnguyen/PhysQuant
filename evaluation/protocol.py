# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Binary messages of JSON values and numpy arrays, shared by the evaluator and policy servers.

A message is a 4-byte big-endian header length, a UTF-8 JSON header, then the raw
array buffers. Arrays in the header are replaced by ``{"__ndarray__": index, "dtype": ..., "shape": ...}``.
Only numpy and the standard library are needed, so model environments can copy this file.
"""

import json
import struct

import numpy as np

PROTOCOL_VERSION = 1


def encode(value):
    """Serialize nested dicts, lists, scalars, strings, and numpy arrays."""
    buffers = []

    def pack(item):
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            buffers.append(array.tobytes())
            return {"__ndarray__": len(buffers) - 1, "dtype": array.dtype.str, "shape": list(array.shape)}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): pack(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [pack(child) for child in item]
        return item

    body = pack(value)
    sizes = [len(buffer) for buffer in buffers]
    header = json.dumps({"version": PROTOCOL_VERSION, "body": body, "sizes": sizes}).encode()
    return struct.pack(">I", len(header)) + header + b"".join(buffers)


def decode(message):
    """Inverse of :func:`encode`; arrays are returned as writable numpy arrays."""
    (length,) = struct.unpack(">I", message[:4])
    header = json.loads(message[4 : 4 + length])
    if header.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version {header.get('version')}")
    offsets, position = [], 4 + length
    for size in header["sizes"]:
        offsets.append((position, position + size))
        position += size

    def unpack(item):
        if isinstance(item, dict) and "__ndarray__" in item:
            start, end = offsets[item["__ndarray__"]]
            return np.frombuffer(message[start:end], dtype=np.dtype(item["dtype"])).reshape(item["shape"]).copy()
        if isinstance(item, dict):
            return {key: unpack(child) for key, child in item.items()}
        if isinstance(item, list):
            return [unpack(child) for child in item]
        return item

    return unpack(header["body"])
