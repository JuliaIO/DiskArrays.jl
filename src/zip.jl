
"""
    DiskZip

Replaces `Zip` for disk arrays, for calling `zip` on disk arrays.

Reads out-of-order over chunks, but `collect`s to the correct order.
Less flexible than `Base.Zip` as it can only zip with other `AbstractArray`.

Note: currently only one of the first two arguments of `zip` must be a disk
array to return `DiskZip`.
"""
struct DiskZip{Is<:Tuple}
    is::Is
end
function DiskZip(As::AbstractArray...)
    map(As) do A
        size(A) == size(first(As)) ||
            throw(DimensionMismatch("Arrays zipped with disk arrays must be the same size"))
    end
    # Get the chunkes of the first Chunked array
    chunks = reduce(As; init=nothing) do acc, A
        if isnothing(acc) && (haschunks(A) isa Chunked)
            eachchunk(A)
        else
            acc
        end
    end
    if isnothing(chunks)
        return DiskZip(As)
    else
        rechunked = map(As) do A
            MockChunkedDiskArray(A, chunks)
        end
        return DiskZip(rechunked)
    end
end

Base.iterate(dz::DiskZip) = Base.iterate(Iterators.Zip(dz.is))
Base.iterate(dz::DiskZip, i) = Base.iterate(Iterators.Zip(dz.is), i)
Base.first(dz::DiskZip) = Base.first(Iterators.Zip(dz.is))
Base.last(dz::DiskZip) = Base.last(Iterators.Zip(dz.is))
Base.length(dz::DiskZip) = Base.length(Iterators.Zip(dz.is))
Base.size(dz::DiskZip) = Base.size(Iterators.Zip(dz.is))
function Base.IteratorSize(::Type{DiskZip{Is}}) where {Is<:Tuple}
    return Base.IteratorSize(Iterators.Zip{Is})
end
function Base.IteratorEltype(::Type{DiskZip{Is}}) where {Is<:Tuple}
    return Base.IteratorEltype(Iterators.Zip{Is})
end

# For now we only allow zip on exact same-sized arrays

# Collect zipped disk arrays in the right order
function Base.collect(dz::DiskZip)
    out = similar(first(dz.is), eltype(dz))
    i = iterate(dz)
    for I in eachindex(first(dz.is))
        out[I] = first(i)
        i = iterate(dz, last(i))
    end
    return out
end

_zip_error() = throw(ArgumentError("Cannot `zip` a disk array with an iterator"))

function Base.zip(A1::AbstractDiskArray, A2::AbstractDiskArray, As::AbstractArray...)
    return DiskZip(A1, A2, As...)
end
function Base.zip(A1::AbstractDiskArray, A2::AbstractArray, As::AbstractArray...)
    return DiskZip(A1, A2, As...)
end
function Base.zip(A1::AbstractArray, A2::AbstractDiskArray, As::AbstractArray...)
    return DiskZip(A1, A2, As...)
end

Base.zip(::AbstractDiskArray, x, xs...) = _zip_error()
Base.zip(x, ::AbstractDiskArray, xs...) = _zip_error()
Base.zip(x::AbstractDiskArray, ::AbstractDiskArray, xs...) = _zip_error()

macro implement_zip(t)
    t = esc(t)
    quote
        Base.zip(A1::$t, A2::$t, As::AbstractArray...) = $DiskZip(A1, A2, As...)
        Base.zip(A1::$t, A2::AbstractArray, As::AbstractArray...) = $DiskZip(A1, A2, As...)
        Base.zip(A1::AbstractArray, A2::$t, As::AbstractArray...) = $DiskZip(A1, A2, As...)

        function Base.zip(A1::AbstractDiskArray, A2::$t, As::AbstractArray...)
            return $DiskZip(A1, A2, As...)
        end
        function Base.zip(A1::$t, A2::AbstractDiskArray, As::AbstractArray...)
            return $DiskZip(A1, A2, As...)
        end

        Base.zip(::$t, x, xs...) = $_zip_error()
        Base.zip(x, ::$t, xs...) = $_zip_error()
        Base.zip(::$t, ::$t, xs...) = $_zip_error()
    end
end
