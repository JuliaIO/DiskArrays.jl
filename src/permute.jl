"""
    AbstractPermutedDiskArray <: AbstractDiskArray

Abstract supertype for diskarray with permuted dimensions.
"""
abstract type AbstractPermutedDiskArray{T,N,perm,iperm,A} <: AbstractDiskArray{T,N} end

"""
    PermutedDiskArray <: AbstractPermutedDiskArray

A lazily permuted disk array returned by `permutedims(diskarray, permutation)`.
"""
struct PermutedDiskArray{T,N,perm,iperm,A<:AbstractArray{T,N}} <: AbstractPermutedDiskArray{T,N,perm,iperm,A}
    parent::A
end
# We use PermutedDimsArray internals instead of duplicating them,
# and just copy the type parameters it calculates.
PermutedDiskArray(A::AbstractArray, perm::Union{Tuple,AbstractVector}) =
    PermutedDiskArray(A, PermutedDimsArray(CartesianIndices(A), perm))
function PermutedDiskArray(
    a::A, ::PermutedDimsArray{<:Any,<:Any,perm,iperm}
) where {A<:AbstractArray{T,N},perm,iperm} where {T,N}
    PermutedDiskArray{T,N,perm,iperm,A}(a)
end

# We need explicit ConstructionBase support as perm and iperm are only in the type.
# We include N so that only arrays of the same dimensionality can be set with this perm and iperm
struct PermutedDiskArrayConstructor{N,perm,iperm} end

(::PermutedDiskArrayConstructor{N,perm,iperm})(a::A) where A<:AbstractArray{T,N} where {T,N,perm,iperm} = 
    PermutedDiskArray{T,N,perm,iperm,A}(a)

ConstructionBase.constructorof(::Type{<:PermutedDiskArray{<:Any,N,perm,iperm}}) where {N,perm,iperm} = 
    PermutedDiskArrayConstructor{N,perm,iperm}()

# Base methods

Base.parent(a::PermutedDiskArray) = a.parent
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

_getperm(::AbstractPermutedDiskArray{<:Any,<:Any,perm}) where {perm} = perm
_getiperm(::AbstractPermutedDiskArray{<:Any,<:Any,<:Any,iperm}) where {iperm} = iperm

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
