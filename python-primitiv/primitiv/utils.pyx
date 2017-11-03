cdef str py_primitiv_utils_encoding = None


def set_encoding(str encoding = None):
    global py_primitiv_utils_encoding
    if encoding:
        py_primitiv_utils_encoding = encoding
    else:
        import locale
        lang, py_primitiv_utils_encoding = locale.getdefaultlocale()


def get_encoding():
    return py_primitiv_utils_encoding


cdef string str_py2cpp(str s):
    return s.encode(py_primitiv_utils_encoding)


cdef str str_cpp2py(string s):
    return s.decode(py_primitiv_utils_encoding)
