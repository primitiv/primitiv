from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv.device cimport wrapDevice, _Device
from primitiv.shape cimport _Shape, normShape
from primitiv.graph cimport _Graph, wrapNode, Node, _Node
from primitiv.parameter cimport _Parameter

cimport numpy as np
import numpy as np

cdef inline vector[float] ndarrays_to_vector(list arrays):
    cdef vector[float] result
    cdef np.float32_t *np_data
    cdef unsigned datasize
    cdef np.ndarray data_tmp
    cdef unsigned j, i
    if len(arrays) == 0:
        raise TypeError("arrays contains no item")
    datasize = arrays[0].size
    shape = arrays[0].shape
    result.resize(len(arrays) * datasize)
    for j, data in enumerate(arrays):
        if shape != data.shape:
            raise TypeError("arrays contains different shaped ndarrays")
        data_tmp = np.array(data, dtype=np.float32, order="F")
        np_data = <np.float32_t *> data_tmp.data
        for i in range(datasize):
            result[j * datasize + i] = np_data[i]
    return result

cdef inline vector[unsigned] ndarray_to_vector_unsigned(np.ndarray array):
    cdef vector[unsigned] result
    cdef unsigned *np_data
    cdef unsigned datasize
    cdef np.ndarray array_tmp

    if array.dtype != np.uint32:
        raise TypeError("numpy.ndarray must be constructed from uint32 data")

    datasize = array.size
    result.resize(datasize)
    array_tmp = np.array(array, order="F")
    np_data = <unsigned*> array_tmp.data
    for i in range(datasize):
        result[i] = np_data[i]
    return result
