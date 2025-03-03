"""
    RechunkedDiskArray <: AbstractDiskArray

    RechunkedDiskArray(parent::AbstractArray, chunks::GridChunks)

A disk array that forces a specific chunk pattern, 
regardless of the true chunk pattern of the parent array.

This is useful in `zip` and other operations that can iterate
over multiple arrays with different patterns.
"""
struct RechunkedDiskArray{T,N,A<:AbstractArray{T,N},C<:GridChunks} <: AbstractDiskArray{T,N}
    parent::A
    chunks::C
end

"""
  rechunk(data::AbstractArray,chunks)

Change the chunks of the underlying DiskArray. Note that this will not change the chunking of the underlying data itself, it will just make the data
"look" like it had a different chunking. If you need a persistent on-disk representation of this chunking, save the resulting array. 

The chunks argument can take one of the following forms:

•  a DiskArrays.GridChunks object

•  a tuple specifying the chunk size along each dimension
"""
rechunk(data::AbstractDiskArray, chunks::GridChunks) = RechunkedDiskArray(data, chunks)

Base.parent(A::RechunkedDiskArray) = A.parent
Base.size(A::RechunkedDiskArray) = size(parent(A))

# DiskArrays interface

haschunks(::RechunkedDiskArray) = Chunked()
eachchunk(A::RechunkedDiskArray) = A.chunks

# These could be more efficient with memory in some cases, but this is simple
readblock!(A::RechunkedDiskArray, data, I...) = _readblock_rechunked(A, data, I...)
readblock!(A::RechunkedDiskArray, data, I::AbstractVector...) =
    _readblock_rechunked(A, data, I...)
writeblock!(A::RechunkedDiskArray, data, I...) = writeblock!(parent(A), data, I...)

function _readblock_rechunked(A, data, I...)
    if haschunks(parent(A)) isa Chunked
        readblock!(parent(A), data, I...)
    else
        # Handle non disk arrays that may be chunked for e.g. chunked `zip`
        copyto!(data, view(parent(A), I...))
    end
end
