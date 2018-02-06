===========================
primitiv File Format v0.1
===========================


**primitiv File Format** is a common binary format to store/load data used in
primitiv.
It uses the `MessagePack <https://msgpack.org/>`_ wire format as the inner
binary representation.


Legend
------


::

    +------+     +---------------+---------------+...
    | Type |  =  | Member Type 1 | Member Type 2 |
    |      |     | Member Name 1 | Member Name 2 |
    +------+     +---------------+---------------+...


Types
-----


::

    +-------+     +---------------+--------+
    | Shape |  =  | array<uint32> | uint32 |
    |       |     | dims          | batch  |
    +-------+     +---------------+--------+

In the current version, the ``batch`` member is always ``1`` for all ``Shape``
objects.

::

    +--------+     +-------+------+
    | Tensor |  =  | Shape | bin  |
    |        |     | shape | data |
    +--------+     +-------+------+

``data`` member has an *array of single-precision floating number* with the
following format:

- Byte order: *Little-endian* (differ than MessagePack's float)
- Array order: *Column-major (Fortran)*
- Batch is treated as the *last dimension* of the shape
  (if ``shape.batch > 1``).
  I.e., The next data begins just after the previous data according to the
  *column-major array order*.

::

    +-----------+     +--------+--------+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+.........
    | Parameter |  =  | Tensor | uint32 | str         | Tensor        |
    |           |     | value  | N      | stat_key[1] | stat_value[1] | N times
    +-----------+     +--------+--------+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+.........

::

    +-------+     +--------+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+.........
    | Model |  =  | uint32 | array<str>   | Parameter      |
    |       |     | N      | param_key[1] | param_value[1] | N times
    +-------+     +--------+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+.........

The key of each parameter represents the *address* of the parameter from the
root model. E.g.:

- ``param_key == ["foo"]``: Parameter has the name ``"foo"``, and is directly
  owned by the *root model*.
- ``param_key == ["foo", "bar"]``: Parameter has the name ``"bar"``, and is
  owned by the submodel ``"foo"``.

::

    +-----------+     +------------------+-----------------+
    | Optimizer |  =  | map<str, uint32> | map<str, float> |
    |           |     | uint_configs     | float_configs   |
    +-----------+     +------------------+-----------------+


File Format
-----------


::

    +-----------+-----------+-----------+----------------------------------------+
    | uint32    | uint32    | uint32    | Shape|Tensor|Parameter|Model|Optimizer |
    | ver_major | ver_minor | data_type | data                                   |
    +-----------+-----------+-----------+----------------------------------------+

Version numbers are typically equal to following:

- ``ver_major == 0``
- ``ver_minor == 1``

Following table shows the correspondence between ``data_type`` and ``data``:

============= =========
data_type     data
============= =========
``0x0``       Shape
``0x100``     Tensor
``0x200``     Parameter
``0x300``     Model
``0x400``     Optimizer
============= =========
