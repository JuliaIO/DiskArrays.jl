"""
    AbstractDiskArray <: AbstractArray

Abstract DiskArray type that can be inherited by Array-like data structures that
have a significant random access overhead and whose access pattern follows
n-dimensional (hyper)-rectangles.
"""
abstract type AbstractDiskArray{T,N} <: AbstractArray{T,N} end

# Base methods

Base.ndims(::AbstractDiskArray{<:Any, N}) where N = N
Base.eltype(::AbstractDiskArray{T}) where T = T
function Base.show(io::IO, ::MIME"text/plain", X::AbstractDiskArray)
    println(io, "Disk Array with size ", join(size(X), " x "))
end
function Base.show(io::IO, X::AbstractDiskArray)
    println(io, "Disk Array with size ", join(size(X), " x "))
end

# DiskArrays interface

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

function readblock!(A::AbstractDiskArray, A_ret, r::AbstractVector...)
    # Implement fallback method if DiskArray does not support strided reading
    # Currently this allocates an intermediate array. In the future, this
    # function could test how sparse the reading is and maybe be smarter here
    # by reading slices.
    mi, ma = map(minimum, r), map(maximum, r)
    A_temp = similar(A_ret, map((a, b)->b-a+1, mi, ma))
    readblock!(A, A_temp, map(:, mi, ma)...)
    A_ret .= view(A_temp, map(ir->ir.-(minimum(ir).-1), r)...)
    return nothing
end

function writeblock!(A::AbstractDiskArray, A_ret, r::AbstractVector...)
    # Implement fallback method if DiskArray does not support strided reading
    # Currently this allocates an intermediate array. In the future, this
    # function could test how sparse the reading is and maybe be smarter here
    # by reading slices.
    mi, ma = map(minimum, r), map(maximum, r)
    A_temp = similar(A_ret, map((a, b)->b-a+1, mi, ma))
    A_temp[map(ir->ir.-(minimum(ir).-1), r)...] = A_ret
    writeblock!(A, A_temp, map(:, mi, ma)...)
    return nothing
end

function getindex_disk(a, i...)
    inds, trans = interpret_indices_disk(a, i)
    data = Array{eltype(a)}(undef, map(length, inds)...)
    readblock!(a, data, inds...)
    return trans(data)
end

function setindex_disk!(a::AbstractDiskArray{T}, v::T, i...) where T<:AbstractArray
    setindex_disk!(a, [v], i...)
end
function setindex_disk!(a::AbstractDiskArray, v::AbstractArray, i...)
    inds, trans = interpret_indices_disk(a, i)
    data = reshape(v, map(length, inds))
    writeblock!(a, data, inds...)
    return v
end

"""
    interpret_indices_disk(A, r)

Function that translates a list of user-supplied indices into plain ranges and
integers for reading blocks of data. This function respects additional indexing
rules like omitting additional trailing indices.

The passed array handle A must implement methods for `Base.size` and `Base.ndims`
The function returns two values:

  1. a tuple whose length equals `ndims(A)` containing only unit
  ranges and integers. This contains the minimal "bounding box" of data that
  has to be read from disk.
  2. A callable object which transforms the hyperrectangle read from disk to
  the actual shape that represents the Base getindex behavior.
"""
function interpret_indices_disk(A, r::Tuple)
    throw(ArgumentError("Indices of type $(typeof(r)) are not yet "))
end
# Read the entire array and reshape to 1D in the end
function interpret_indices_disk(A, ::Tuple{Colon})
    return map(Base.OneTo, size(A)), Reshaper(prod(size(A)))
end
interpret_indices_disk(A, r::Tuple{<:CartesianIndex}) =
interpret_indices_disk(A, r[1].I)
interpret_indices_disk(A, r::Tuple{<:CartesianIndices}) =
interpret_indices_disk(A, r[1].indices)
function interpret_indices_disk(A, r::NTuple{N,Union{Integer,AbstractVector,Colon}}) where N
    if ndims(A) == N
        inds = map(_convert_index, r, size(A))
        resh = DimsDropper(findints(r))
        return inds, resh
    elseif ndims(A) < N
        foreach((ndims(A) + 1):N) do i
            r[i] == 1 || throw(BoundsError(A, r))
        end
        _, rshort = commonlength(size(A), r)
        return interpret_indices_disk(A, rshort)
    else
        size(A, N + 1) == 1 || throw(BoundsError(A, r))
        return interpret_indices_disk(A, (r..., 1))
    end
end
function interpret_indices_disk(A, r::Tuple{<:AbstractArray{<:Bool}})
    ba = r[1]
    if ndims(A) == ndims(ba)
        inds = getbb(ba)
        resh = a -> a[view(ba, inds...)]
        return inds, resh
    elseif ndims(ba)==1
        return interpret_indices_disk(A, (reshape(ba, size(A)), ))
    else
        throw(BoundsError(A, r))
    end
end
function interpret_indices_disk(A, r::NTuple{1,AbstractVector})
    lininds = first(r)
    cartinds = CartesianIndices(A)
    mi, ma = extrema(view(cartinds, lininds))
    inds = map((i1, i2) -> i1:i2, mi.I, ma.I)
    resh = a -> map(lininds) do ii
        a[cartinds[ii] - mi + oneunit(mi)]
    end
    return inds, resh
end

# Reshaping functors

struct Reshaper{I}
    reshape_indices::I
end
(r::Reshaper)(a) = reshape(a, r.reshape_indices)

struct DimsDropper{D}
    d::D
end
(d::DimsDropper)(a) = length(d.d)==ndims(a) ? a[1] : dropdims(a, dims=d.d)

struct TransformStack{S}
    s::S
end
(s::TransformStack)(a) = ∘(s.s...)(a)

function getbb(ar::AbstractArray{Bool})
    maxval = CartesianIndex(size(ar))
    minval = CartesianIndex{ndims(ar)}()
    function reduceop(i1, i2)
        i2 === nothing ? i1 : (min(i1[1], i2), max(i1[2], i2))
    end
    mi, ma = mapfoldl(reduceop, zip(CartesianIndices(ar), ar), init=(maxval, minval)) do ii
        ind, val = ii
        val ? ind : nothing
    end
    return map((i1, i2) -> i1:i2, mi.I, ma.I)
end

# Helper functions

"For two given tuples return a truncated version of both so they have common length"
commonlength(a, b) = _commonlength((first(a), ), (first(b), ), Base.tail(a), Base.tail(b))
commonlength(::Tuple{}, b) = (), ()
commonlength(a, ::Tuple{}) = (), ()

_commonlength(a1, b1, a, b) = _commonlength((a1..., first(a)), (b1..., first(b)), Base.tail(a), Base.tail(b))
_commonlength(a1, b1, ::Tuple{}, b) = (a1, b1)
_commonlength(a1, b1, a, ::Tuple{}) = (a1, b1)

"Find the indices of elements containing integers in a Tuple"
findints(x) = _findints((), 1, x...)

_findints(c, i, x::Integer, rest...) = _findints((c..., i), i + 1, rest...)
_findints(c, i, x, rest...) = _findints(c, i + 1, rest...)
_findints(c, i)  = c
# Normal indexing for a full subset of an array
_convert_index(i::Integer, s::Integer) = i:i
_convert_index(i::AbstractVector, s::Integer) = i
_convert_index(::Colon, s::Integer) = Base.OneTo(Int(s))

# Implementation macros

macro implement_getindex(t)
    quote
        function Base.getindex(a::$t, i...)
            getindex_disk(a, i...)
        end
    end
end

macro implement_setindex(t)
    quote
        Base.setindex!(a::$t, v::AbstractArray, i...) = setindex_disk!(a, v, i...)
        # Add an extra method if a single number is given
        Base.setindex!(a::$t{<:Any,N}, v, i...) where N =
            setindex!(a, fill(v, ntuple(i->1, N)...), i...)
        # Special care must be taken for logical indexing, we can not avoid reading
        # the data before writing
        function Base.setindex!(a::$t, v::AbstractArray, i::AbstractArray{<:Bool})
            inds, trans = interpret_indices_disk(a, (i, ))
            data = Array{eltype(a)}(undef, map(length, inds)...)
            readblock!(a, data, inds...)

            data = reshape(v, map(length, inds))
            writeblock!(a, data, inds...)
            return v
        end
    end
end

