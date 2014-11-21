var searchIndex = {};
searchIndex['ndarray'] = {"items":[[0,"","ndarray","The **ndarray** crate provides the [**Array**](./struct.Array.html) type, an\nn-dimensional numerical container similar to numpy's ndarray.\n"],[1,"Si","","A slice, a description of a range of an array axis."],[1,"Indexes","","An iterator of the indexes of an array shape."],[1,"Array","","The **Array** type is an *N-dimensional array*."],[1,"Elements","","An iterator over the elements of an array."],[1,"ElementsMut","","An iterator over the elements of an array."],[1,"IndexedElements","","An iterator over the indexes and elements of an array."],[1,"IndexedElementsMut","","An iterator over the indexes and elements of an array."],[3,"d1","","Construct one-dimensional array shape. Helper function to use where\ninteger literal inference isn't working."],[3,"d2","","Construct two-dimensional array shape. Helper function to use where\ninteger literal inference isn't working."],[3,"d3","","Construct three-dimensional array shape. Helper function to use where\ninteger literal inference isn't working."],[3,"d4","","Construct four-dimensional array shape. Helper function to use where\ninteger literal inference isn't working."],[3,"ixrange","","Like `range`, except with array indexes."],[3,"arr0","","Return a zero-dimensional array with the element `x`."],[3,"arr1","","Return a one-dimensional array with elements from `xs`."],[3,"arr2","","Return a two-dimensional array with elements from `xs`."],[0,"linalg","","A few linear algebra operations on two-dimensional arrays."],[3,"eye","ndarray::linalg","Return the identity matrix of dimension *n*."],[3,"least_squares","","Solve *a x = b* with linear least squares approximation."],[3,"cholesky","","Factor *a = L L<sup>T</sup>*."],[3,"subst_fw","","Solve *L x = b* where *L* is a lower triangular matrix."],[3,"subst_bw","","Solve *U x = b* where *U* is an upper triangular matrix."],[4,"Col","","Column vector."],[4,"Mat","","Rectangular matrix."],[6,"Ring","","Trait union for a ring with 1."],[6,"Field","","Trait union for a field."],[6,"ComplexField","","A real or complex number."],[10,"conjugate","","",0],[9,"sqrt_real","","",0],[10,"is_complex","","",0],[10,"conjugate","num::complex","",1],[10,"sqrt_real","","",1],[10,"is_complex","","",1],[10,"index","ndarray","",2],[10,"index_mut","","",2],[10,"eq","","Return `true` if the array shapes and all elements of `self` and\n`other` are equal. Return `false` otherwise.",2],[10,"from_iter","","",2],[10,"hash","","",2],[10,"encode","","",2],[10,"decode","","",2],[10,"fmt","","Format the array using `Show` and apply the formatting parameters used\nto each element.",2],[10,"fmt","","Format the array using `LowerExp` and apply the formatting parameters used\nto each element.",2],[10,"fmt","","Format the array using `UpperExp` and apply the formatting parameters used\nto each element.",2],[10,"ndim","collections::vec","",3],[10,"slice","","",3],[10,"slice_mut","","",3],[10,"fmt","ndarray","",4],[10,"hash","","",4],[10,"eq","","",4],[10,"ne","","",4],[10,"clone","","",4],[10,"clone","","",5],[10,"new","","Create an iterator over the array shape `dim`.",5],[10,"new1","","Create an iterator over the array shape `a`.",5],[10,"new2","","Create an iterator over the array shape `(a, b)`.",5],[10,"new3","","Create an iterator over the array shape `(a, b, c)`.",5],[10,"next","","",5],[10,"size_hint","","",5],[10,"clone","","",6],[10,"next","","",6],[10,"size_hint","","",6],[10,"next_back","","",6],[10,"clone","","",7],[10,"next","","",7],[10,"size_hint","","",7],[10,"next","","",8],[10,"size_hint","","",8],[10,"next_back","","",8],[10,"next","","",9],[10,"size_hint","","",9],[4,"Ix","","Array index type"],[4,"Ixs","","Array index type (signed)"],[5,"S","","Slice value for the full range of an axis."],[6,"Dimension","","Trait for the shape and index types of arrays."],[9,"ndim","","",10],[10,"slice","","",10],[10,"slice_mut","","",10],[10,"size","","",10],[10,"default_strides","","",10],[10,"first_index","","",10],[10,"next_for","","Iteration -- Use self as size, and return next index after `index`\nor None if there are no more.",10],[10,"stride_offset","","Return stride offset for index.",10],[10,"stride_offset_checked","","Return stride offset for this dimension and index.",10],[10,"do_slices","","Modify dimension, strides and return data pointer offset",10],[6,"RemoveAxis","","Helper trait to define a larger-than relation for array shapes:\nremoving one axis from *Self* gives smaller dimension *E*."],[9,"remove_axis","","",11],[10,"clone","","",2],[10,"zeros","","Construct an Array with zeros.",2],[10,"from_elem","","Construct an Array with copies of `elem`.",2],[10,"from_vec","","Create a one-dimensional array from a vector (no allocation needed).",2],[10,"from_iter","","Create a one-dimensional array from an iterator.",2],[10,"range","","Create a one-dimensional Array from interval `[begin, end)`",2],[10,"from_vec_dim","","Create an array from a vector (with no allocation needed).",2],[10,"dim","","Return the shape of the array.",2],[10,"shape","","Return the shape of the array as a slice.",2],[10,"is_standard_layout","","Return `true` if the array data is laid out in\ncontiguous “C order” where the last index is the most rapidly\nvarying.",2],[10,"raw_data","","Return a slice of the array's backing data in memory order.",2],[10,"slice","","Return a sliced array.",2],[10,"islice","","Slice the array's view in place.",2],[10,"slice_iter","","Return an iterator over a sliced view.",2],[10,"at","","Return a reference to the element at `index`, or return `None`\nif the index is out of bounds.",2],[10,"uchk_at","","Perform *unchecked* array indexing.",2],[10,"uchk_at_mut","","Perform *unchecked* array indexing.",2],[10,"iter","","Return an iterator of references to the elements of the array.",2],[10,"indexed_iter","","Return an iterator of references to the elements of the array.",2],[10,"isubview","","Collapse dimension `axis` into length one,\nand select the subview of `index` along that axis.",2],[10,"broadcast_iter","","Act like a larger size and/or shape array by *broadcasting*\ninto a larger shape, if possible.",2],[10,"swap_axes","","Swap axes `ax` and `bx`.",2],[10,"diag_iter","","Return an iterator over the diagonal elements of the array.",2],[10,"diag","","Return the diagonal as a one-dimensional array.",2],[10,"map","","Apply `f` elementwise and return a new array with\nthe results.",2],[10,"subview","","Select the subview `index` along `axis` and return an\narray with that axis removed.",2],[10,"ensure_unique","","Make the array unshared.",2],[10,"at_mut","","Return a mutable reference to the element at `index`, or return `None`\nif the index is out of bounds.",2],[10,"iter_mut","","Return an iterator of mutable references to the elements of the array.",2],[10,"indexed_iter_mut","","Return an iterator of indexes and mutable references to the elements of the array.",2],[10,"slice_iter_mut","","Return an iterator of mutable references into the sliced view\nof the array.",2],[10,"sub_iter_mut","","Select the subview `index` along `axis` and return an iterator\nof the subview.",2],[10,"diag_iter_mut","","Return an iterator over the diagonal elements of the array.",2],[10,"raw_data_mut","","Return a mutable slice of the array's backing data in memory order.",2],[10,"reshape","","Transform the array into `shape`; any other shape\nwith the same number of elements is accepted.",2],[10,"assign","","Perform an elementwise assigment to `self` from `other`.",2],[10,"assign_scalar","","Perform an elementwise assigment to `self` from scalar `x`.",2],[10,"sum","","Return sum along `axis`.",2],[10,"mean","","Return mean along `axis`.",2],[10,"row_iter","","Return an iterator over the elements of row `index`.",2],[10,"col_iter","","Return an iterator over the elements of column `index`.",2],[10,"mat_mul","","Perform matrix multiplication of rectangular arrays `self` and `other`.",2],[10,"mat_mul_col","","Perform the matrix multiplication of the rectangular array `self` and\ncolumn vector `other`.",2],[10,"allclose","","Return `true` if the arrays' elementwise differences are all within\nthe given absolute tolerance.<br>\nReturn `false` otherwise, or if the shapes disagree.",2],[10,"iadd","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"iadd_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"add","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"isub","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"isub_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"sub","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"imul","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"imul_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"mul","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"idiv","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"idiv_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"div","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"irem","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"irem_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"rem","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"ibitand","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"ibitand_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"bitand","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"ibitor","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"ibitor_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"bitor","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"ibitxor","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"ibitxor_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"bitxor","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"ishl","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"ishl_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"shl","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"ishr","","Perform an elementwise arithmetic operation between `self` and `other`,\n*in place*.",2],[10,"ishr_scalar","","Perform an elementwise arithmetic operation between `self` and the scalar `x`,\n*in place*.",2],[10,"shr","","Perform an elementwise arithmetic operation between `self` and `other`,\nand return the result.",2],[10,"ineg","","Perform an elementwise negation of `self`, *in place*.",2],[10,"neg","","Perform an elementwise negation of `self` and return the result.",2],[10,"inot","","Perform an elementwise unary not of `self`, *in place*.",2],[10,"not","","Perform an elementwise unary not of `self` and return the result.",2],[10,"slice","","",10],[10,"slice_mut","","",10],[10,"size","","",10],[10,"default_strides","","",10],[10,"first_index","","",10],[10,"next_for","","Iteration -- Use self as size, and return next index after `index`\nor None if there are no more.",10],[10,"stride_offset","","Return stride offset for index.",10],[10,"stride_offset_checked","","Return stride offset for this dimension and index.",10],[10,"do_slices","","Modify dimension, strides and return data pointer offset",10],[10,"ndim","","",12],[10,"size","","",12],[10,"first_index","","",12],[10,"next_for","","",12],[10,"stride_offset","","Self is an index, return the stride offset",12],[10,"stride_offset_checked","","Return stride offset for this dimension and index.",12],[10,"remove_axis","","",12]],"paths":[[6,"ComplexField"],[1,"Complex"],[1,"Array"],[1,"Vec"],[1,"Si"],[1,"Indexes"],[1,"Elements"],[1,"IndexedElements"],[1,"ElementsMut"],[1,"IndexedElementsMut"],[6,"Dimension"],[6,"RemoveAxis"],[4,"Ix"]]};
initSearch(searchIndex);
