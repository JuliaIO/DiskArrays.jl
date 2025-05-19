"""
    SubDiskArray <: AbstractDiskArray

Abstract supertype for a view of an AbstractDiskArray
"""
abstract type AbstractSubDiskArray{T,N,P,I,L} <: AbstractDiskArray{T,N} end

"""
    SubDiskArray <: AbstractDiskArray

A replacement for `Base.SubArray` for disk arrays, returned by `view`.
"""
struct SubDiskArray{T,N,P,I,L} <: AbstractSubDiskArray{T,N,P,I,L}
    v::SubArray{T,N,P,I,L}
end

# Base methods
subarray(a::SubDiskArray) = a.v
function Base.view(a::T, i...) where T<:AbstractSubDiskArray 
    basetype = Base.typename(T).wrapper
    basetype(view(subarray(a), i...))
end
Base.view(a::AbstractSubDiskArray, i::CartesianIndices) = view(a, i.indices...)
Base.size(a::AbstractSubDiskArray) = size(subarray(a))
Base.parent(a::AbstractSubDiskArray) = parent(subarray(a))
Base.parentindices(a::AbstractSubDiskArray) = parentindices(subarray(a))

_replace_colon(s, ::Colon) = Base.OneTo(s)
_replace_colon(s, r) = r

# Diskarrays.jl interface
function readblock!(a::AbstractSubDiskArray, aout, i::OrdinalRange...)
    pinds = parentindices(view(a, i...))
    getindex_disk!(aout, parent(a), pinds...)
end
function writeblock!(a::AbstractSubDiskArray, v, i::OrdinalRange...)
    pinds = parentindices(view(a, i...))
    setindex_disk!(parent(a), v, pinds...)
end
haschunks(a::AbstractSubDiskArray) = haschunks(parent(a))
eachchunk(a::AbstractSubDiskArray) = eachchunk_view(haschunks(parent(a)), a)

function eachchunk_view(::Chunked, vv)
    pinds = parentindices(vv)
    if any(ind -> !isa(ind, Union{Int,AbstractRange,Colon,AbstractVector{<:Integer}}), pinds)
        throw(ArgumentError("Unable to determine chunksize for view of type $(typeof.(pinds))."))
    end
    chunksparent = eachchunk(parent(vv))
    newchunks = map(chunksparent.chunks, pinds) do ch, pi
        pi isa Integer ? nothing : subsetchunks(ch, pi)
    end
    filteredchunks = reduce(newchunks; init=()) do acc, x
        isnothing(x) ? acc : (acc..., x)
    end
    return GridChunks(filteredchunks...)
end
eachchunk_view(::Unchunked, a) = estimate_chunksize(a)

function view_disk(A, I...)
    @inline
    # Modified from Base.view
    J = to_indices(A, I)
    @boundscheck checkbounds(A, J...)
    J′ = Base.rm_singleton_indices(ntuple(Returns(true), Val(ndims(A))), J...)
    SubDiskArray(Base.unsafe_view(A, J′...))
end

# Implementaion macro
macro implement_subarray(t)
    t = esc(t)
    quote
        @inline Base.view(a::$t, i...) = view_disk(a, i...)
        Base.vec(a::$t) = view(a, :)
    end
end
