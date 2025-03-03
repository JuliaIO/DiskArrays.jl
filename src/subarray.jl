"""
    SubDiskArray <: AbstractDiskArray

A replacement for `Base.SubArray` for disk arrays, returned by `view`.
"""
struct SubDiskArray{T,N,P,I,L} <: AbstractDiskArray{T,N}
    v::SubArray{T,N,P,I,L}
end

# Base methods
Base.view(a::SubDiskArray, i...) = SubDiskArray(view(a.v, i...))
Base.view(a::SubDiskArray, i::CartesianIndices) = view(a, i.indices...)
Base.size(a::SubDiskArray) = size(a.v)
Base.parent(a::SubDiskArray) = a.v.parent

_replace_colon(s, ::Colon) = Base.OneTo(s)
_replace_colon(s, r) = r

# Diskarrays.jl interface
function readblock!(a::SubDiskArray, aout, i::OrdinalRange...)
    pinds = parentindices(view(a.v, i...))
    getindex_disk!(aout, parent(a.v), pinds...)
end
function writeblock!(a::SubDiskArray, v, i::OrdinalRange...)
    pinds = parentindices(view(a.v, i...))
    setindex_disk!(parent(a.v), v, pinds...)
end
haschunks(a::SubDiskArray) = haschunks(parent(a.v))
eachchunk(a::SubDiskArray) = eachchunk_view(haschunks(a.v.parent), a.v)

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

# Implementaion macro

macro implement_subarray(t)
    t = esc(t)
    quote
        function Base.view(a::$t, i...)
            i2 = _replace_colon.(size(a), i)
            return SubDiskArray(SubArray(a, i2))
        end
        Base.view(a::$t, i::CartesianIndices) = view(a, i.indices...)
        Base.vec(a::$t) = view(a, :)
    end
end
