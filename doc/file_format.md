*primitiv* File Format v0.1
===========================

Primitiv file format is using [MessagePack](https://msgpack.org/)
wire format as the inner binary representation.

Legends
-------

```
+------+     +---------------+---------------+...
| Type |  =  | Member Type 1 | Member Type 2 |
|      |     | Member name 1 | Member name 2 |
+------+     +---------------+---------------+...
```

Defined Types
-------------

```
+-------+     +---------------+--------+
| Shape |  =  | array<uint32> | uint32 |
|       |     | dims          | batch  |
+-------+     +---------------+--------+

+--------+     +-------+------+
| Tensor |  =  | Shape | bin  |
|        |     | shape | data |
+--------+     +-------+------+

+-----------+     +--------+------------------+
| Parameter |  =  | Tensor | map<str, Tensor> |
|           |     | value  | stats            |
+-----------+     +--------+------------------+

+-------+     +----------------------------+
| Model |  =  | map<array<str>, Parameter> |
|       |     | params                     |
+-------+     +----------------------------+
```

File Format
-----------

```
+-----------+------------+-----------+------------------------------+
| uint32    | uint32     | uint32    | Shape|Tensor|Parameter|Model |
| ver_major | ver_minor  | data_type | data                         |
+-----------+------------+-----------+------------------------------+
```
