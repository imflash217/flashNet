```
__editor__: @imflash217
__copyright__: flash.ai @ 2019
__book__: Guide to Numpy by Oliphant
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

# `broadcasting`
1. `ufunc`s are always element-by-element operation
2. Broadcasting allows `ufunc`s to deal in a meaningful way with inputs that do not have exactly the same shape.

Rules of `broadcasting`:
1. `Rule #1`: If all input arrays do not have the same number of dimensions then, a `1` will be
    repeatedly pre-pended to the shapes of the smaller arrays untill all arrays have the same number of dimensions.
2. `Rule #2`: Arrays with size of `1` along a particular dimension act as if they had
    the size of the largest shape along that dimension. The value of the array element is
    assumed to be same along that dimension for the `broadcast` array.
3. `Rule #3`: After application of the broadcasting rules, all arrays must have the same shape.

* `Notes`:One important aspect of broadcasting is the calculation of functions on the regularly spaced grids.

# `np.ndarray` attributes:
1. Array attributes reflect information about the array that are intrinsic to the array itself.
2. The exposed attributes are the core parts of the array.
3. Only some of the attributes can be reset meaningfully without creating a new array.
4. Following are the attributes of `np.ndarray`:
    * `flags`   : settable = NO     : provides info about how the memory area used for array is to be interpreted.
    * `shape`   : settable = YES    : tuple showing the array shape; setting this attribute reshapes the array.
    * `strides` : settable = YES    : tuple showing how many _bytes_ must be jumped in to get to next element.
    * `ndim`    : settable = NO     : number of dimensions in the array.
    * `data`    : settable = YES    : buffer object loosely wrapping the array data (only for single segment array).
    * `size`    : settable = NO     : number of elements in the array.
    * `itemsize`: settable = NO     : size (in _bytes_) of each element in the array.
    * `nbytes`  : settable = NO     : total number of bytes used.
    * `base`    : settable = NO     : object this array is using for its data buffer. `None` if it owns its memory.
    * `dtype`   : settable = YES    : the data-type object of the array.
    * `real`    : settable = YES    : real part of the array. Setting copies data to the real part of the array.
    * `imag`    : settable = YES    : imaginary-part or 0-only array (if type is not complex). Setting works only if type is complex.
    * `flat`    : settable = YES    : one-dimensional, indexible iterator object that somwhat acts like 1D array.
    * `ctypes`  : settable = NO     : object to simplify the simplify the interaction of this object with `ctypes` module.
    * `__array_interface__`         : dictionary with keys for compliance with Python side of array protocol
    * `__array_struct__`            : array interface on C-level
    * `__array_priority__`          : always 0 for `base` type of `ndarray`

5.
