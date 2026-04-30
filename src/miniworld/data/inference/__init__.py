"""Inference-time Batch construction (no LMDB-backed structure dataset).

Build a ``Batch`` directly from user-supplied fasta + a3m + (optional) template
+ contacts, without going through ``CIFMolAttached`` / cif LMDB.
"""

from .build import build_inference_batch
from .spec import InferenceSpec

__all__ = ["InferenceSpec", "build_inference_batch"]
