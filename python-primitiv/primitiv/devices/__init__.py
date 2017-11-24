try:
    from primitiv.devices._cuda_device import _CUDA as CUDA
#except ModuleNotFoundError:
except ImportError:
    pass

try:
    from primitiv.devices._opencl_device import _OpenCL as OpenCL
#except ModuleNotFoundError:
except ImportError:
    pass

from primitiv.devices._naive_device import _Naive as Naive

__all__ = [
    "CUDA",
    "Naive",
    "OpenCL",
]
