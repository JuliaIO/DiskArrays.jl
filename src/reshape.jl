import Base: _throw_dmrs
import Base.PermutedDimsArrays: genperm

"""
    AbstractReshapedDiskArray <: AbstractDiskArray

Abstract supertype for a replacements of `Base.ReshapedArray` for `AbstractDiskArray`s`
"""
abstract type AbstractReshapedDiskArray{T,N,P,M} <: AbstractDiskArray{T,N} end

"""
    ReshapedDiskArray <: AbstractDiskArray

A replacement for `Base.ReshapedArray` for disk arrays,
returned by `reshape`.


Reshaping is really not trivial, because the access pattern would 
completely change for reshaped arrays, rectangles would not remain 
rectangles in the parent array. 

However, we can support the case where only singleton dimensions are added, 
later we could allow more special cases like joining two dimensions to one
"""
struct ReshapedDiskArray{T,N,P<:AbstractArray{T},M} <: AbstractReshapedDiskArray{T,N,P,M}
    parent::P
    keepdim::NTuple{M,Int}
    newsize::NTuple{N,Int}
end

# Base methods
Base.size(r::AbstractReshapedDiskArray) = r.newsize
Base.parent(r::AbstractReshapedDiskArray) = r.parent
keepdim(r::AbstractReshapedDiskArray) = r.keepdim

# DiskArrays interface

haschunks(a::AbstractReshapedDiskArray) = haschunks(parent(a))
function eachchunk(a::AbstractReshapedDiskArray{<:Any,N}) where {N}
    pchunks = eachchunk(parent(a))
    inow::Int = 0
    outchunks = ntuple(N) do idim
        if in(idim, keepdim(a))
            inow += 1
            pchunks.chunks[inow]
        else
            RegularChunks(1, 0, size(a, idim))
        end
    end
    return GridChunks(outchunks...)
end
function DiskArrays.readblock!(a::AbstractReshapedDiskArray, aout, i::OrdinalRange...)
    inew = tuple_tuple_getindex(i, keepdim(a))
    DiskArrays.readblock!(parent(a), reshape(aout, map(length, inew)), inew...)
    return nothing
end
function DiskArrays.writeblock!(a::AbstractReshapedDiskArray, v, i::OrdinalRange...)
    inew = tuple_tuple_getindex(i, keepdim(a))
    DiskArrays.writeblock!(parent(a), reshape(v, map(length, inew)), inew...)
    return nothing
end
function reshape_disk(parent, dims)
    n = length(parent)
    ndims(parent) > length(dims) &&
        error("For DiskArrays, reshape is restricted to adding singleton dimensions")
    prod(dims) == n || _throw_dmrs(n, "size", dims)
    ipassed::Int = 0
    keepdim = map(size(parent)) do s
        while true
            ipassed += 1
            d = dims[ipassed]
            if d > 1
                d != s && error(
                    "For DiskArrays, reshape is restricted to adding singleton dimensions",
                )
                return ipassed
            else
                # For existing trailing 1s
                d == s == 1 && return ipassed
            end
        end
    end
    return ReshapedDiskArray{eltype(parent),length(dims),typeof(parent),ndims(parent)}(
        parent, keepdim, dims
    )
end

tuple_tuple_getindex(t, i) = _ttgi((), t, i...)
_ttgi(o, t, i1, irest...) = _ttgi((o..., t[i1]), t, irest...)
_ttgi(o, t, i1) = (o..., t[i1])

# Implementaion macro
macro implement_reshape(t)
    t = esc(t)
    quote
        function Base._reshape(A::$t, dims::NTuple{N,Int}) where {N}
            return reshape_disk(A, dims)
        end
    end
end

# For ambiguity
function Base._reshape(A::AbstractDiskArray{<:Any,1}, dims::Tuple{Int64})
    return reshape_disk(A, dims)
end
