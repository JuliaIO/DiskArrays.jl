"""
    PermutedDiskArray <: AbstractDiskArray

A lazily permuted disk array returned by `permutedims(diskarray, permutation)`.
"""
struct PermutedDiskArray{T,N,perm,iperm,A<:AbstractArray{T,N}} <: AbstractDiskArray{T,N}
    a::A
end
PermutedDiskArray(A::AbstractArray, perm::Union{Tuple,AbstractVector}) =
    PermutedDiskArray(A, PermutedDimsArray(CartesianIndices(A), perm))
# We use PermutedDimsArray internals instead of duplicating them,
# and just copy the type parameters
function PermutedDiskArray(
    a::A, perm::PermutedDimsArray{<:Any,<:Any,perm,iperm}
) where {A<:AbstractArray{T,N},perm,iperm} where {T,N} =
    PermutedDiskArray{T,N,perm,iperm,A}(a)
end

# Base methods

Base.size(a::PermutedDiskArray) = genperm(size(parent(a)), _getperm(a))

# DiskArrays interface

haschunks(a::PermutedDiskArray) = haschunks(parent(a))
function eachchunk(a::PermutedDiskArray)
    # Get the parent chunks
    gridchunks = eachchunk(parent(a))
    perm = _getperm(a)
    # Return permuted GridChunks
    return GridChunks(genperm(gridchunks.chunks, perm)...)
end
function DiskArrays.readblock!(a::PermutedDiskArray, aout, i::OrdinalRange...)
    iperm = _getiperm(a)
    # Permute the indices
    inew = genperm(i, iperm)
    # Permute the dest block and read from the true parent
    DiskArrays.readblock!(a.a.parent, PermutedDimsArray(aout, iperm), inew...)
    return nothing
end
function DiskArrays.writeblock!(a::PermutedDiskArray, v, i::OrdinalRange...)
    iperm = _getiperm(a)
    inew = genperm(i, iperm)
    # Permute the dest block and write from the true parent
    DiskArrays.writeblock!(a.a.parent, PermutedDimsArray(v, iperm), inew...)
    return nothing
end

_getperm(::PermutedDiskArray{<:Any,<:Any,perm}) where {perm} = perm
_getiperm(::PermutedDiskArray{<:Any,<:Any,<:Any,iperm}) where {iperm} = iperm

# Implementation macro

macro implement_permutedims(t)
    t = esc(t)
    quote
        Base.permutedims(parent::$t, perm) = PermutedDiskArray(parent, perm)
        # It's not correct to return a PermutedDiskArray from the PermutedDimsArray constructor.
        # Instead we need a Base julia method that behaves like view for SubArray, such as `lazypermutedims`.
        # But until that exists this is better than returning a broken disk array.
        Base.PermutedDimsArray(parent::$t, perm) = PermutedDiskArray(parent, perm)
    end
end
