"""
    PermutedDiskArray <: AbstractDiskArray

A lazily permuted disk array returned by `permutedims(diskarray, permutation)`.
"""
struct PermutedDiskArray{T,N,P<:PermutedDimsArray{T,N}} <: AbstractDiskArray{T,N}
    a::P
end

# Base methods

Base.size(a::PermutedDiskArray) = size(a.a)

# DiskArrays interface

haschunks(a::PermutedDiskArray) = haschunks(a.a.parent)
function eachchunk(a::PermutedDiskArray)
    # Get the parent chunks
    gridchunks = eachchunk(a.a.parent)
    perm = _getperm(a.a)
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

_getperm(a::PermutedDiskArray) = _getperm(a.a)
_getperm(::PermutedDimsArray{<:Any,<:Any,perm}) where {perm} = perm

_getiperm(a::PermutedDiskArray) = _getiperm(a.a)
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
