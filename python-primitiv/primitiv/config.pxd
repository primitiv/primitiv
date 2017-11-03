from libcpp.string cimport string


cdef string pystr_to_cppstr(str s)
cdef str cppstr_to_pystr(string s)
