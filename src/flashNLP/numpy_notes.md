```
__editor__: @imflash217
__copyright__: flash.ai @ 2019
```

# `np.ndarray`
1. Two important features of `np.ndarray` are:
    * `shape`
    * `dtype`
2. The `dtype` objects are flexible enough to contain references to arrays of other data
    types and therefore can be used to define nested records. Reference: `np.recarray` subclass.
3. Python typically defines only one data-type of a particular data class

    Eg. one integer type, one floating-point type etc.

    But Numpy provides 21 different data-type-descriptors builtin which are based on
    the types available in C & Cython programming language.
4. Some types like `str_`, `unicode_` and `void` are extremely flexible i.e they can have
    items that are of arbitray size.
5. The `object_` is special because arrays with `dtype="O"` don't return an array scalar
    on item access but instead return the actual object refeenced in the array.
6. Data-type hierarchy:
    * `generic`:
        * `bool_`
        * `object_`
        * number:
            * integer:
                * signedinteger:
                    * `byte`
                    * `short`
                    * `intc`
                    * `int_`
                    * `longlong`
                * unsignedinteger:
                    * `ubyte`
                    * `ushort`
                    * `uintc`
                    * `unit`
                    * `ulonglong`
            * inexact:
                * floating:
                    * `single`
                    * `float`
                    * `longfloat`
                * complexfloating:
                    * `csingle`
                    * `complex`
                    * `clongfloat`
        * flexible:
            * `void`
            * character:
                * `str_`
                * `unicode_`

# Basic Indexing (slicing)


