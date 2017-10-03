from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv.device cimport wrapDevice, _Device
from primitiv.shape cimport _Shape, normShape
from primitiv.graph cimport _Graph, wrapNode, Node, _Node
from primitiv.parameter cimport _Parameter

cimport numpy as np
import numpy as np

cdef inline vector[float] ndarray_to_vector(np.ndarray array):
    cdef vector[float] result
    cdef np.float32_t *np_data
    cdef unsigned datasize
    cdef np.ndarray array_tmp

    datasize = array.size
    result.resize(datasize)
    array_tmp = np.array(array, order="F")
    np_data = <np.float32_t *> array_tmp.data
    for i in range(datasize):
        result[i] = np_data[i]
    return result


cdef inline vector[unsigned] ndarray_to_vector_unsigned(np.ndarray array):
    cdef vector[unsigned] result
    cdef unsigned *np_data
    cdef unsigned datasize
    cdef np.ndarray array_tmp

    datasize = array.size
    result.resize(datasize)
    array_tmp = np.array(array, order="F")
    np_data = <unsigned*> array_tmp.data
    for i in range(datasize):
        result[i] = np_data[i]
    return result
