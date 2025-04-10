"""
    AbstractPermutedDiskArray <: AbstractDiskArray

Abstract supertype for diskarray with permuted dimensions.
"""
abstract type AbstractPermutedDiskArray{T,N,P<:PermutedDimsArray{T,N}} <: AbstractDiskArray{T,N} end

"""
    PermutedDiskArray <: AbstractPermutedDiskArray

A lazily permuted disk array returned by `permutedims(diskarray, permutation)`.
"""
struct PermutedDiskArray{T,N,P<:PermutedDimsArray{T,N}} <: AbstractPermutedDiskArray{T,N,P}
    a::P
end

# Base methods
Base.size(a::AbstractPermutedDiskArray) = size(a.a)
Base.parent(a::AbstractPermutedDiskArray) = a.a.parent
# DiskArrays interface

haschunks(a::AbstractPermutedDiskArray) = haschunks(parent(a))
function eachchunk(a::AbstractPermutedDiskArray)
    # Get the parent chunks
    gridchunks = eachchunk(parent(a))
    perm = _getperm(a)
    # Return permuted GridChunks
    return GridChunks(genperm(gridchunks.chunks, perm)...)
end
function DiskArrays.readblock!(a::AbstractPermutedDiskArray, aout, i::OrdinalRange...)
    iperm = _getiperm(a)
    # Permute the indices
    inew = genperm(i, iperm)
    # Permute the dest block and read from the true parent
    DiskArrays.readblock!(parent(a), PermutedDimsArray(aout, iperm), inew...)
    return nothing
end
function DiskArrays.writeblock!(a::AbstractPermutedDiskArray, v, i::OrdinalRange...)
    iperm = _getiperm(a)
    inew = genperm(i, iperm)
    # Permute the dest block and write from the true parent
    DiskArrays.writeblock!(parent(a), PermutedDimsArray(v, iperm), inew...)
    return nothing
end

_getperm(a::AbstractPermutedDiskArray) = _getperm(a.a)
_getperm(::PermutedDimsArray{<:Any,<:Any,perm}) where {perm} = perm

_getiperm(a::AbstractPermutedDiskArray) = _getiperm(a.a)
_getiperm(::PermutedDimsArray{<:Any,<:Any,<:Any,iperm}) where {iperm} = iperm

# Implementaion macros

function permutedims_disk(a, perm)
    pd = PermutedDimsArray(a, perm)
    return PermutedDiskArray{eltype(a),ndims(a),typeof(pd)}(pd)
end

macro implement_permutedims(t)
    t = esc(t)
    quote
        Base.permutedims(parent::$t, perm) = permutedims_disk(parent, perm)
    end
end
