import Base: _throw_dmrs

"""
    AbstractReshapedDiskArray <: AbstractDiskArray

Abstract supertype for a replacements of `Base.ReshapedArray` for `AbstractDiskArray`s`
"""
abstract type AbstractReshapedDiskArray{T,N,P,M} <: AbstractDiskArray{T,N} end

"""
    ReshapedDiskArray <: AbstractReshapedDiskArray

A replacement for `Base.ReshapedArray` for disk arrays,
returned by `reshape`.

Reshaping is really not trivial, because the access pattern would 
completely change for reshaped arrays, rectangles would not remain 
rectangles in the parent array. 

However, we can support the case where only singleton dimensions are added, 
later we could allow more special cases like joining two dimensions to one
"""
struct ReshapedDiskArray{T,N,P<:AbstractArray{T},DMAP} <: AbstractReshapedDiskArray{T,N,P,DMAP}
    parent::P
    dmap::Val{DMAP}
    newsize::NTuple{N,Int}
end
dmap(::ReshapedDiskArray{<:Any,<:Any,<:Any,DMAP}) where DMAP = DMAP
# Base methods
Base.size(r::AbstractReshapedDiskArray) = r.newsize
Base.parent(r::AbstractReshapedDiskArray) = r.parent


# DiskArrays interface
haschunks(a::AbstractReshapedDiskArray) = haschunks(parent(a))
function eachchunk(a::AbstractReshapedDiskArray{<:Any,N}) where {N}
    pchunks = eachchunk(parent(a)).chunks
    dm = dmap(a)
    cnew = ntuple(i -> dm[i] == -1 ? RegularChunks(1, 0, 1) : pchunks[dm[i]], N)
    return GridChunks(cnew...)
end
function DiskArrays.readblock!(a::AbstractReshapedDiskArray, aout, i::OrdinalRange...)
    inew = reshape_index(a, 1:1, i)
    DiskArrays.readblock!(parent(a), reshape(aout, map(length, inew)), inew...)
    return nothing
end
function DiskArrays.writeblock!(a::AbstractReshapedDiskArray, v, i::OrdinalRange...)
    inew = reshape_index(a, 1:1, i)
    DiskArrays.writeblock!(parent(a), reshape(v, map(length, inew)), inew...)
    return nothing
end
function reshape_disk(parent, dims)
    n = length(parent)
    prod(dims) == n || DiskArrays._throw_dmrs(n, "size", dims)
    iparent::Int = 1
    dmap = map(dims) do d
        while true
            s = size(parent, iparent)
            if d > 1
                if s == 1
                    #We are removing a singleton dimension
                    iparent += 1
                    continue
                elseif d != s
                    error(
                        "For DiskArrays, reshape is restricted to adding or removing singleton dimensions",
                    )
                else
                    iparent += 1
                    return iparent - 1
                end
            else
                return -1
            end
        end
    end
    return ReshapedDiskArray{eltype(parent),length(dmap),typeof(parent),dmap}(
        parent, Val(dmap), dims
    )
end

function reshape_index(a, default, replace)
    inew = map(_ -> default, size(parent(a)))
    dm = dmap(a)
    for ii in eachindex(dm)
        m = dm[ii]
        if m != -1
            indx = replace[ii]
            inew = Base.setindex(inew, indx, m)
        end
    end
    inew
end


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
