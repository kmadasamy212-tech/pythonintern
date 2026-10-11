"""
Generated gRPC stubs package for distributed LLM inference.
"""

import sys
from pathlib import Path

# Ensure proto_gen folder is in sys.path so generated stubs resolve peer imports
_pkg_dir = str(Path(__file__).parent.resolve())
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from . import distributed_inference_pb2 as pb2
from . import distributed_inference_pb2_grpc as pb2_grpc

__all__ = ["pb2", "pb2_grpc"]
