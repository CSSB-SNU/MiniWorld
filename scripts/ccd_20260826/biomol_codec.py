"""Byte-exact reimplementation of biomol.core.utils.{to_bytes,load_bytes}.

Copied field-for-field from biomol 1.0.1 so records written here are
indistinguishable from those already in preprocessed_CCD.lmdb.
"""

import json
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
from zstandard import ZstdCompressor, ZstdDecompressor


def to_bytes(data: Mapping[str, Any], level: int = 6) -> bytes:
    """Serialize a dict containing NumPy arrays to zstd-compressed bytes."""

    def _flatten_data(data):
        template = {}
        flatten = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                _key = str(id(value))
                template[key] = _key
                buffer = BytesIO()
                np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
                flatten[_key] = buffer.getbuffer()
            elif isinstance(value, dict):
                _template, _flatten = _flatten_data(value)
                template[key] = _template
                flatten.update(_flatten)
            else:
                template[key] = value
        return template, flatten

    template, flatten_data = _flatten_data(data)
    header = {
        "template": template,
        "arrays": {key: len(value) for key, value in flatten_data.items()},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    output = BytesIO()
    with ZstdCompressor(level=level).stream_writer(output, closefd=False) as writer:
        writer.write(len(header_bytes).to_bytes(8, "little"))
        writer.write(header_bytes)
        for key in flatten_data:
            writer.write(flatten_data[key])
    return output.getvalue()


def load_bytes(byte_data: bytes) -> dict:
    """Deserialize zstd-compressed bytes back into a dict."""

    def _reconstruct_data(template, flatten):
        data = {}
        for key, value in template.items():
            if isinstance(value, str) and value in flatten:
                data[key] = np.load(BytesIO(flatten[value]), allow_pickle=False)
            elif isinstance(value, dict):
                data[key] = _reconstruct_data(value, flatten)
            else:
                data[key] = value
        return data

    with ZstdDecompressor().stream_reader(BytesIO(byte_data)) as reader:
        raw = reader.read()
    hlen = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
    payload = memoryview(raw)[8 + hlen :]

    offset = 0
    flatten = {}
    for key, ln in header["arrays"].items():
        flatten[key] = bytes(payload[offset : offset + ln])
        offset += ln
    return _reconstruct_data(header["template"], flatten)
