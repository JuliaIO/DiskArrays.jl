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
    chunkexists(a, chunkidxs::Integer...) -> Bool
    chunkexists(a, chunkidxs::Tuple{Vararg{Integer}}) -> Bool
    chunkexists(a, chunkidx::Union{CartesianIndex,ChunkIndex}) -> Bool
    chunkexists(a, indices) -> AbstractArray{Bool}

Return whether a chunk is stored. Chunk coordinates are one-based indices into
[`eachchunk(a)`](@ref eachchunk), with one coordinate per array dimension.
Callers must supply valid chunk coordinates; the fallback assumes every chunk
exists and returns `true` without checking bounds or reading data.

Backends with optional chunk storage can specialize
`chunkexists(a::CustomDiskArray, chunkidxs::Integer...)`. A stored chunk containing
only fill values still exists; an absent chunk may read as fill values.

An iterable of coordinates queries multiple chunks, returning a Boolean array
in iteration order and preserving the shape of array inputs. Each coordinate
may be an integer tuple, `CartesianIndex`, or [`ChunkIndex`](@ref).
The fallback calls scalar `chunkexists` for each coordinate. Backends can
specialize the iterable form, for example with `indices::AbstractVector{<:Tuple}`,
to batch storage queries.
"""
chunkexists(a, chunkidxs::Integer...) = true
chunkexists(a, chunkidxs::Tuple{Vararg{Integer}}) = chunkexists(a, chunkidxs...)
chunkexists(a, chunkidx::CartesianIndex) = chunkexists(a, Tuple(chunkidx)...)
chunkexists(a, chunkidx::ChunkIndex) = chunkexists(a, chunkidx.I)
chunkexists(a, indices) = Bool[chunkexists(a, i) for i in indices]

"""
    haschunks(a)

Returns a trait for the chunk pattern of a dis array, 
[`Chunked`](@ref) or [`Unchunked`](@ref).
"""
function haschunks end
haschunks(x) = Unchunked()

function Base.checkbounds(::Type{Bool}, a::AbstractDiskArray, i::ChunkIndex)
    checkbounds(Bool, eachchunk(a), i.I)
end
