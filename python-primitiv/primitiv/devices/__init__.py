from primitiv.devices._naive_device import Naive
__all__ = ["Naive"]

try:
    from primitiv.devices._cuda_device import CUDA
    __all__.append("CUDA")
#except ModuleNotFoundError:
except ImportError:
    pass

try:
    from primitiv.devices._opencl_device import OpenCL
    __all__.append("OpenCL")
#except ModuleNotFoundError:
except ImportError:
    pass

