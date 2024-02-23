var documenterSearchIndex = {"docs":
[{"location":"","page":"DiskArrays","title":"DiskArrays","text":"CurrentModule = DiskArrays","category":"page"},{"location":"#DiskArrays","page":"DiskArrays","title":"DiskArrays","text":"","category":"section"},{"location":"","page":"DiskArrays","title":"DiskArrays","text":"Modules = [DiskArrays, DiskArrays.TestTypes]","category":"page"},{"location":"#DiskArrays.DiskArrays","page":"DiskArrays","title":"DiskArrays.DiskArrays","text":"DiskArrays.jl\n\n(Image: Lifecycle) (Image: Stable Docs) (Image: Dev Docs) (Image: CI) (Image: Codecov)\n\nThis package provides a collection of utilities for working with n-dimensional array-like data structures that do have considerable overhead for single read operations.  Most important examples are arrays that represent data on hard disk that are accessed through a C library or that are compressed in chunks.  It can be inadvisable to make these arrays a direct subtype of AbstractArray many functions working with AbstractArrays assume fast random access into single values (including basic things like getindex, show, reduce, etc...). \n\nCurrently supported features are:\n\ngetindex/setindex with the same rules as base (trailing or singleton dimensions etc)\nviews into DiskArrays\na fallback Base.show method that does not call getindex repeatedly\nimplementations for mapreduce and mapreducedim, that respect the chunking of the underlying\n\ndataset. This greatly increases performance of higher-level reductions like sum(a,dims=d)\n\nan iterator over the values of a DiskArray that caches a chunk of data and returns the values\n\nwithin. This allows efficient usage of e.g. using DataStructures; counter(a)\n\ncustomization of broadcast when there is a DiskArray on the LHS. This at least makes things\n\nlike a.=5 possible and relatively fast\n\nAbstractDiskArray Interface definition\n\nPackage authors who want to use this library to make their disk-based array an AbstractDiskArray should at least implement methods for the following functions:\n\nBase.size(A::CustomDiskArray)\nreadblock!(A::CustomDiskArray{T,N},aout,r::Vargarg{AbstractUnitRange,N})\nwriteblock!(A::CustomDiskArray{T,N},ain,r::Vargarg{AbstractUnitRange,N})\n\nHere readblock! will read a subset of array A in a hyper-rectangle defined by the unit ranges r. The results shall be written into aout. writeblock! should write the data given by ain into the (hyper-)rectangle of A defined by r When defining the functions it can be safely assumed that length(r) == ndims(A) as well as size(ain) == length.(r). However, bounds checking is not performed by the DiskArray machinery and currently should be done by the implementation. \n\nIf the data on disk has rectangular chunks as underlying storage units, you should addtionally implement the following methods to optimize some operations like broadcast, reductions and sparse indexing:\n\nDiskArrays.haschunks(A::CustomDiskArray) = DiskArrays.Chunked()\nDiskArrays.eachchunk(A::CustomDiskArray) = DiskArrays.GridChunks(A, chunksize)\n\nwhere chunksize is a int-tuple of chunk lengths. If the array does not have an internal chunking structure, one should define\n\nDiskArrays.haschunks(A::CustomDiskArray) = DiskArrays.Unchunked()\n\nImplementing only these methods makes all kinds of strange indexing patterns work (Colons, StepRanges, Integer vectors, Boolean masks, CartesianIndices, Arrays of CartesianIndex, and mixtures of all these) while making sure that as few readblock! or writeblock! calls as possible are performed by reading a rectangular bounding box of the required array values and re-arranging the resulting values into the output array. \n\nIn addition, DiskArrays.jl provides a few optimizations for sparse indexing patterns to avoid reading and discarding  too much unnecessary data from disk, for example for indices like A[:,:,[1,1500]]. \n\nExample\n\nHere we define a new array type that wraps a normal AbstractArray. The only access method that we define is a readblock! function where indices are strictly given as unit ranges along every dimension of the array. This is a very common API used in libraries like HDF5, NetCDF and Zarr. We also define a chunking, which will control the way iteration and reductions are computed. In order to understand how exactly data is accessed, we added the additional print statements in the readblock! and writeblock! functions.\n\nusing DiskArrays\n\nstruct PseudoDiskArray{T,N,A<:AbstractArray{T,N}} <: AbstractDiskArray{T,N}\n  parent::A\n  chunksize::NTuple{N,Int}\nend\nPseudoDiskArray(a;chunksize=size(a)) = PseudoDiskArray(a,chunksize)\nhaschunks(a::PseudoDiskArray) = Chunked()\neachchunk(a::PseudoDiskArray) = GridChunks(a,a.chunksize)\nBase.size(a::PseudoDiskArray) = size(a.parent)\nfunction DiskArrays.readblock!(a::PseudoDiskArray,aout,i::AbstractUnitRange...)\n  ndims(a) == length(i) || error(\"Number of indices is not correct\")\n  all(r->isa(r,AbstractUnitRange),i) || error(\"Not all indices are unit ranges\")\n  println(\"Reading at index \", join(string.(i),\" \"))\n  aout .= a.parent[i...]\nend\nfunction DiskArrays.writeblock!(a::PseudoDiskArray,v,i::AbstractUnitRange...)\n  ndims(a) == length(i) || error(\"Number of indices is not correct\")\n  all(r->isa(r,AbstractUnitRange),i) || error(\"Not all indices are unit ranges\")\n  println(\"Writing to indices \", join(string.(i),\" \"))\n  view(a.parent,i...) .= v\nend\na = PseudoDiskArray(rand(4,5,1))\n\nDisk Array with size 10 x 9 x 1\n\nNow all the Base indexing behaviors work for our array, while minimizing the number of reads that have to be done:\n\na[:,3]\n\nReading at index Base.OneTo(10) 3:3 1:1\n\n10-element Array{Float64,1}:\n 0.8821177068878834\n 0.6220977650963209\n 0.22676949571723437\n 0.3177934541451004\n 0.08014908894614026\n 0.9989838001681182\n 0.5865160181790519\n 0.27931778627456216\n 0.449108677620097  \n 0.22886146620923808\n\nAs can be seen from the read message, only a single call to readblock is performed, which will map to a single call into the underlying C library.\n\nmask = falses(4,5,1)\nmask[3,2:4,1] .= true\na[mask]\n\n3-element Array{Int64,1}:\n 6\n 7\n 8\n\nOne can check in a similar way, that reductions respect the chunks defined by the data type:\n\nsum(a,dims=(1,3))\n\nReading at index 1:5 1:3 1:1\nReading at index 6:10 1:3 1:1\nReading at index 1:5 4:6 1:1\nReading at index 6:10 4:6 1:1\nReading at index 1:5 7:9 1:1\nReading at index 6:10 7:9 1:1\n\n1×9×1 Array{Float64,3}:\n[:, :, 1] =\n 6.33221  4.91877  3.98709  4.18658  …  6.01844  5.03799  3.91565  6.06882\n ````\n\nWhen a DiskArray is on the LHS of a broadcasting expression, the results with be\nwritten chunk by chunk:\n\n\njulia va = view(a,5:10,5:8,1) va .= 2.0 a[:,:,1]\n\n\n\nWriting to indices 5:5 5:6 1:1 Writing to indices 6:10 5:6 1:1 Writing to indices 5:5 7:8 1:1 Writing to indices 6:10 7:8 1:1 Reading at index Base.OneTo(10) Base.OneTo(9) 1:1\n\n10×9 Array{Float64,2}:  0.929979   0.664717  0.617594  0.720272   …  0.564644  0.430036  0.791838  0.392748   0.508902  0.941583  0.854843      0.682924  0.323496  0.389914  0.761131   0.937071  0.805167  0.951293      0.630261  0.290144  0.534721  0.332388   0.914568  0.497409  0.471007      0.470808  0.726594  0.97107  0.251657   0.24236   0.866905  0.669599      2.0       2.0       0.427387  0.388476   0.121011  0.738621  0.304039   …  2.0       2.0       0.687802  0.991391   0.621701  0.210167  0.129159      2.0       2.0       0.733581  0.371857   0.549601  0.289447  0.509249      2.0       2.0       0.920333  0.76309    0.648815  0.632453  0.623295      2.0       2.0       0.387723  0.0882056  0.842403  0.147516  0.0562536     2.0       2.0       0.107673 ````\n\nAccessing strided Arrays\n\nThere are situations where one wants to read every other value along a certain axis or provide arbitrary strides. Some DiskArray backends may want to provide optimized methods to read these strided arrays.  In this case a backend can define readblock!(a,aout,r::OrdinalRange...) and the respective writeblock method which will overwrite the fallback behavior that would read the whol block of data and only return the desired range.\n\nArrays that do not implement eachchunk\n\nThere are arrays that live on disk but which are not split into rectangular chunks, so that the haschunks trait returns Unchunked(). In order to still enable broadcasting and reductions for these arrays, a chunk size will be estimated in a way that a certain memory limit per chunk is not exceeded. This memory limit defaults to 100MB and can be modified by changing DiskArrays.default_chunk_size[]. Then a chunk size is computed based on the element size of the array. However, there are cases where the size of the element type is undefined, e.g. for Strings or variable-length vectors. In these cases one can overload the DiskArrays.element_size function for certain container types which returns an approximate element size (in bytes). Otherwise the size of an element will simply be assumed to equal the value stored in DiskArrays.fallback_element_size which defaults to 100 bytes. \n\n[ci-img]: https://github.com/meggart/DiskArrays.jl/workflows/CI/badge.svg [ci-url]: https://github.com/meggart/DiskArrays.jl/actions?query=workflow%3ACI [codecov-img]: http://codecov.io/github/meggart/DiskArrays.jl/coverage.svg?branch=main [codecov-url]: (http://codecov.io/github/meggart/DiskArrays.jl?branch=main)\n\n\n\n\n\n","category":"module"},{"location":"#DiskArrays.default_chunk_size","page":"DiskArrays","title":"DiskArrays.default_chunk_size","text":"The target chunk size for processing for unchunked arrays in MB, defaults to 100MB\n\n\n\n\n\n","category":"constant"},{"location":"#DiskArrays.fallback_element_size","page":"DiskArrays","title":"DiskArrays.fallback_element_size","text":"A fallback element size for arrays to determine a where elements have unknown size like strings. Defaults to 100MB\n\n\n\n\n\n","category":"constant"},{"location":"#DiskArrays.AbstractDiskArray","page":"DiskArrays","title":"DiskArrays.AbstractDiskArray","text":"AbstractDiskArray <: AbstractArray\n\nAbstract DiskArray type that can be inherited by Array-like data structures that have a significant random access overhead and whose access pattern follows n-dimensional (hyper)-rectangles.\n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.CachedDiskArray","page":"DiskArrays","title":"DiskArrays.CachedDiskArray","text":"CachedDiskArray <: AbstractDiskArray\n\nCachedDiskArray(A::AbstractArray; maxsize=1000)\n\nWrap some disk array A with a caching mechanism that will  keep chunks up to a total of maxsize megabytes, dropping the least used chunks when maxsize is exceeded.\n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.ChunkIndex","page":"DiskArrays","title":"DiskArrays.ChunkIndex","text":"ChunkIndex{N}\n\nThis can be used in indexing operations when one wants to extract a full data chunk from a DiskArray. Useful for  iterating over chunks of data. d[ChunkIndex(1,1)] will extract the first chunk of a 2D-DiskArray\n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.ChunkIndices","page":"DiskArrays","title":"DiskArrays.ChunkIndices","text":"ChunkIndices{N}\n\nRepresents an iterator \n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.ConcatDiskArray","page":"DiskArrays","title":"DiskArrays.ConcatDiskArray","text":"ConcatDiskArray <: AbstractDiskArray\n\nJoins multiple AbstractArrays or AbstractDiskArrays in lazy concatination.\n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.IrregularChunks","page":"DiskArrays","title":"DiskArrays.IrregularChunks","text":"IrregularChunks <: ChunkType\n\nDefines chunks along a dimension where chunk sizes are not constant but arbitrary\n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.IrregularChunks-Tuple{}","page":"DiskArrays","title":"DiskArrays.IrregularChunks","text":"IrregularChunks(; chunksizes)\n\nReturns an IrregularChunks object for the given list of chunk sizes\n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.RegularChunks","page":"DiskArrays","title":"DiskArrays.RegularChunks","text":"RegularChunks <: ChunkType\n\nDefines chunking along a dimension where the chunks have constant size and a potential offset for the first chunk. The last chunk is truncated to fit the array size. \n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.allow_scalar-Tuple{Bool}","page":"DiskArrays","title":"DiskArrays.allow_scalar","text":"allow_scalar(x::Bool)\n\nSpecify if a disk array can do scalar indexing, (with all Int arguments).\n\nSetting allow_scalar(false) can help identify the cause of poor performance.\n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.approx_chunksize-Tuple{DiskArrays.GridChunks}","page":"DiskArrays","title":"DiskArrays.approx_chunksize","text":"approx_chunksize(g::GridChunks)\n\nReturns the aproximate chunk size of the grid. For the dimension with regular chunks, this will be the exact chunk size while for dimensions with irregular chunks this is the average chunks size. Useful for downstream applications that want to distribute computations and want to know about chunk sizes. \n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.arraysize_from_chunksize-Tuple{DiskArrays.RegularChunks}","page":"DiskArrays","title":"DiskArrays.arraysize_from_chunksize","text":"arraysize_from_chunksize(g::ChunkType)\n\nReturns the size of the dimension represented by a chunk object. \n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.cache-Tuple{AbstractArray}","page":"DiskArrays","title":"DiskArrays.cache","text":"cache(A::AbstractArray; maxsize=1000)\n\nWrap internal disk arrays with CacheDiskArray.\n\nThis function is intended to be extended by package that want to re-wrap the disk array afterwards, such as YAXArrays.jl or Rasters.jl.\n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.can_scalar-Tuple{}","page":"DiskArrays","title":"DiskArrays.can_scalar","text":"can_scalar()\n\nCheck if DiskArrays is set to allow scalar indexing, with allow_scalar.\n\nReturns a Bool.\n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.chunktype_from_chunksizes-Tuple{Any}","page":"DiskArrays","title":"DiskArrays.chunktype_from_chunksizes","text":"chunktype_from_chunksizes(chunks)\n\nUtility function that constructs either a RegularChunks or an IrregularChunks object based on a vector of chunk sizes given as worted Integers. Wherever possible it will try to create a regular chunks object.  \n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.eachchunk","page":"DiskArrays","title":"DiskArrays.eachchunk","text":"eachchunk(a)\n\nReturns an iterator with CartesianIndices elements that mark the index range of each chunk within an array.\n\n\n\n\n\n","category":"function"},{"location":"#DiskArrays.element_size-Tuple{AbstractArray}","page":"DiskArrays","title":"DiskArrays.element_size","text":"element_size(a::AbstractArray)\n\nReturns the approximate size of an element of a in bytes. This falls back to calling sizeof on  the element type or to the value stored in DiskArrays.fallback_element_size. Methods can be added for  custom containers. \n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.findints-Tuple{Any}","page":"DiskArrays","title":"DiskArrays.findints","text":"Find the indices of elements containing integers in a Tuple\n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.grid_offset-Tuple{DiskArrays.GridChunks}","page":"DiskArrays","title":"DiskArrays.grid_offset","text":"grid_offset(g::GridChunks)\n\nReturns the offset of the grid for the first chunks. Expect this value to be non-zero for views into regular-gridded arrays. Useful for downstream applications that want to distribute computations and want to know about chunk sizes. \n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.max_chunksize-Tuple{DiskArrays.GridChunks}","page":"DiskArrays","title":"DiskArrays.max_chunksize","text":"max_chunksize(g::GridChunks)\n\nReturns the maximum chunk size of an array for each dimension. Useful for pre-allocating arrays to make sure they can hold a chunk of data. \n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.readblock!","page":"DiskArrays","title":"DiskArrays.readblock!","text":"readblock!(A::AbstractDiskArray, A_ret, r::AbstractUnitRange...)\n\nThe only function that should be implemented by a AbstractDiskArray. This function\n\n\n\n\n\n","category":"function"},{"location":"#DiskArrays.resolve_indices-Tuple{Any, Any}","page":"DiskArrays","title":"DiskArrays.resolve_indices","text":"resolve_indices(a, i)\n\nDetermines a list of tuples used to perform the read or write operations. The returned values are:\n\noutsize size of the output array\ntemp_size size of the temp array passed to readblock\noutput_indices indices for copying into the output array\ntemp_indices indices for reading from temp array\ndata_indices indices for reading from data array\n\n\n\n\n\n","category":"method"},{"location":"#DiskArrays.writeblock!","page":"DiskArrays","title":"DiskArrays.writeblock!","text":"writeblock!(A::AbstractDiskArray, A_in, r::AbstractUnitRange...)\n\nFunction that should be implemented by a AbstractDiskArray if write operations should be supported as well.\n\n\n\n\n\n","category":"function"},{"location":"#DiskArrays.TestTypes.AccessCountDiskArray","page":"DiskArrays","title":"DiskArrays.TestTypes.AccessCountDiskArray","text":"AccessCountDiskArray(A; chunksize)\n\nAn array that counts getindex and setindex calls, to debug and optimise chunk access.\n\ngetindex_count(A) and setindex_count(A) can be used to check the the counters.\n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.TestTypes.ChunkedDiskArray","page":"DiskArrays","title":"DiskArrays.TestTypes.ChunkedDiskArray","text":"ChunkedDiskArray(A; chunksize)\n\nA generic AbstractDiskArray that can wrap any other AbstractArray, with custom chunksize.\n\n\n\n\n\n","category":"type"},{"location":"#DiskArrays.TestTypes.UnchunkedDiskArray","page":"DiskArrays","title":"DiskArrays.TestTypes.UnchunkedDiskArray","text":"UnchunkedDiskArray(A)\n\nA disk array without chunking, that can wrap any other AbstractArray.\n\n\n\n\n\n","category":"type"}]
}
