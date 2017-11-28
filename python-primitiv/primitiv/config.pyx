cdef str py_primitiv_utils_encoding = None


def set_encoding(str encoding = None):
    """Set text encoding for converting unicode in Python.

    :param encoding: String to identify the encoding.
    :type encoding: str

    """
    global py_primitiv_utils_encoding
    if encoding is None:
        import locale
        _, encoding = locale.getdefaultlocale()
        if encoding is None:
            # NOTE(vbkaisetsu):
            # Sometimes, the locale does not have an encoding information. (e.g. LANG=C)
            # In that case, the default encoding is set to "utf-8".
            encoding = "utf-8"
    py_primitiv_utils_encoding = encoding


def get_encoding():
    """Get text encoding for converting unicode in Python.

    :return: Current encoding.
    :rtype: str

    """
    return py_primitiv_utils_encoding


cdef string pystr_to_cppstr(str s):
    return s.encode(py_primitiv_utils_encoding)


cdef str cppstr_to_pystr(string s):
    return s.decode(py_primitiv_utils_encoding)
