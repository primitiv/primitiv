from primitiv._device cimport _Device, Device, wrapDevice
from primitiv._graph cimport _Graph, Graph, wrapGraph


cdef class _DefaultScopeDevice(object):

    def __init__(self, _Device obj = None):
        self.obj = obj

    def __enter__(self):
        # WARNING: DO NOT OVERWRITE self.obj
        # A wrapped object is used in C++.
        # If you overwrite it, the object will be deleted.
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        if self.obj is None:
            self.wrapped_newed = new DefaultScope[Device]()
        else:
            self.wrapped_newed = new DefaultScope[Device]((<_Device> self.obj).wrapped[0])
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __exit__(self, exc_type, exc_value, traceback):
        if self.wrapped_newed is not NULL:
            del self.wrapped_newed
        self.wrapped_newed = NULL
        return False

    def __dealloc__(self):
        if self.wrapped_newed is not NULL:
            del self.wrapped_newed
        self.wrapped_newed = NULL

    @staticmethod
    def get():
        return wrapDevice(&DefaultScopeDevice_get())

    @staticmethod
    def size():
        return DefaultScopeDevice_size()


cdef class _DefaultScopeGraph(object):

    def __init__(self, _Graph obj = None):
        self.obj = obj

    def __enter__(self):
        # WARNING: DO NOT OVERWRITE self.obj
        # A wrapped object is used in C++.
        # If you overwrite it, the object will be deleted.
        if self.wrapped_newed is not NULL:
            raise MemoryError()
        if self.obj is None:
            self.wrapped_newed = new DefaultScope[Graph]()
        else:
            self.wrapped_newed = new DefaultScope[Graph]((<_Graph> self.obj).wrapped[0])
        if self.wrapped_newed is NULL:
            raise MemoryError()
        self.wrapped = self.wrapped_newed

    def __exit__(self, exc_type, exc_value, traceback):
        if self.wrapped_newed is not NULL:
            del self.wrapped_newed
        self.wrapped_newed = NULL
        return False

    def __dealloc__(self):
        if self.wrapped_newed is not NULL:
            del self.wrapped_newed
        self.wrapped_newed = NULL

    @staticmethod
    def get():
        return wrapGraph(&DefaultScopeGraph_get())

    @staticmethod
    def size():
        return DefaultScopeGraph_size()


def _DefaultScope(x):
    if isinstance(x, _Device):
        return _DefaultScopeDevice(x)
    elif isinstance(x, _Graph):
        return _DefaultScopeGraph(x)
    else:
        raise TypeError("Invalid argument")
