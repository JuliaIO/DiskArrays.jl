"""
    AbstractDiskArray <: AbstractArray

Abstract DiskArray type that can be inherited by Array-like data structures that
have a significant random access overhead and whose access pattern follows
n-dimensional (hyper)-rectangles.
"""
abstract type AbstractDiskArray{T,N} <: AbstractArray{T,N} end

"""
    isdisk(a::AbstractArray)

Return `true` if `a` is a `AbstractDiskArray` or follows 
the DiskArrays.jl interface via macros. Otherwise `false`.
"""
isdisk(a::AbstractArray) = isdisk(typeof(a))
isdisk(::Type{<:AbstractArray}) = false

"""
    readblock!(A::AbstractDiskArray, A_ret, r::AbstractUnitRange...)

The only function that should be implemented by a `AbstractDiskArray`. This function
"""
function readblock! end

"""
    writeblock!(A::AbstractDiskArray, A_in, r::AbstractUnitRange...)

Function that should be implemented by a `AbstractDiskArray` if write operations
should be supported as well.
"""
function writeblock! end

"""
    eachchunk(a)

Returns an iterator with `CartesianIndices` elements that mark the index range of each chunk within an array.
"""
function eachchunk end
# Here we implement a fallback chunking for a DiskArray although this should normally
# be over-ridden by the package that implements the interface
eachchunk(a::AbstractArray) = estimate_chunksize(a)

"""
    haschunks(a)

Returns a trait for the chunk pattern of a dis array, 
[`Chunked`](@ref) or [`Unchunked`](@ref).
"""
function haschunks end
haschunks(x) = Unchunked()
