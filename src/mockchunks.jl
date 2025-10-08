"""
    MockChunkedDiskArray <: AbstractDiskArray

    MockChunkedDiskArray(parent::AbstractArray, chunks::GridChunks)

A disk array that pretends to have a specific chunk pattern, 
regardless of the true chunk pattern of the parent array.

This is useful in `zip` and other operations that can iterate
over multiple arrays with different patterns.
"""
struct MockChunkedDiskArray{T,N,A<:AbstractArray{T,N},C<:GridChunks} <: AbstractDiskArray{T,N}
    parent::A
    chunks::C
end

"""
  mockchunks(data::AbstractArray,chunks)

Change the chunk pattern of the underlying DiskArray according to `chunks`. 

Note that this will not change the chunking of the underlying data itself, it will just make the data
"look" like it had a different chunking. If you need a persistent on-disk representation of this chunking, save the resulting array. 

The chunks argument can take one of the following forms:

- a [`DiskArrays.GridChunks`](@ref) object
- a tuple specifying the chunk size along each dimension, like `(10, 10, 1)` for a 3-D array
"""
function mockchunks(data::AbstractArray, chunks)
    gridchunks = if chunks isa GridChunks
        chunks
    else
        GridChunks(data, chunks)
    end
    MockChunkedDiskArray(data, gridchunks)
end


Base.parent(A::MockChunkedDiskArray) = A.parent
Base.size(A::MockChunkedDiskArray) = size(parent(A))

# DiskArrays interface

haschunks(::MockChunkedDiskArray) = Chunked()
eachchunk(A::MockChunkedDiskArray) = A.chunks

# These could be more efficient with memory in some cases, but this is simple
readblock!(A::MockChunkedDiskArray, data, I...) = _readblock_mockchunked(A, data, I...)
readblock!(A::MockChunkedDiskArray, data, I::AbstractVector...) =
    _readblock_mockchunked(A, data, I...)
writeblock!(A::MockChunkedDiskArray, data, I...) = writeblock!(parent(A), data, I...)

function _readblock_mockchunked(A, data, I...)
    if haschunks(parent(A)) isa Chunked
        readblock!(parent(A), data, I...)
    else
        # Handle non disk arrays that may be chunked for e.g. chunked `zip`
        copyto!(data, view(parent(A), I...))
    end
end

Base.@deprecate_binding RechunkedDiskArray MockChunkedDiskArray
