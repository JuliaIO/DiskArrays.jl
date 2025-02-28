"""
    PaddedDiskArray <: AbstractDiskArray

    PaddedDiskArray(A, padding; fill=zero(eltype(A)))

An `AbstractDiskArray` that adds padding to the edges of the parent array.
This can help changing chunk offsets or padding a larger than memory array
before windowing operations.

# Arguments

- `A`: The parent disk array.
- `padding`: A tuple of `Int` lower and upper padding tuples, one for each dimension.

# Keywords

- `fill=zero(eltype(A))`: The value to pad the array with.
"""
struct PaddedDiskArray{T,N,A<:AbstractArray{T,N},C,F<:T} <: AbstractDiskArray{T,N}
    parent::A
    padding::NTuple{N,Tuple{Int,Int}}
    fill::F
    chunks::C
end
function PaddedDiskArray(A::AbstractArray{T,N}, padding::NTuple{N,Tuple{Int,Int}};
    fill=zero(eltype(A)),
) where {T,N}
    map(padding) do (l, u)
        (l < 0 || u < 0) && throw(ArgumentError("Padding must be non-negative"))
    end
    chunks = GridChunks(map(_pad_offset, eachchunk(A).chunks, padding))
    PaddedDiskArray(A, padding, fill, chunks)
end

function _pad_offset(c::RegularChunks, (low, high))
    chunksize = c.chunksize
    # Handle lower padding larger than chunksize
    offset = if low == 0
        c.offset
    else
        c.offset - low + chunksize * (div(low - 1, chunksize) + 1)
    end
    size = c.arraysize + low + high
    return RegularChunks(chunksize, offset, size)
end
function _pad_offset(c::IrregularChunks, (low, high))
    nlowchunks = Int(low > 0) 
    nhighchunks = Int(high > 0)
    offsets = Vector{Int}(undef, length(c.offsets) + nlowchunks + nhighchunks)
    # First offset is always zero
    offsets[begin] = 0
    # Increase original offsets by lower padding
    for (i, o) in enumerate(c.offsets)
        offsets[i + nlowchunks] = o + low
    end
    # Add offset for start of upper padding
    if nhighchunks > 0
        offsets[end] = offsets[end-1] + high
    end
    return IrregularChunks(offsets)
end

Base.parent(A::PaddedDiskArray) = A.parent
function Base.size(A::PaddedDiskArray)
    map(size(parent(A)), A.padding) do s, (low, high)
        s + low + high
    end
end

haschunks(A::PaddedDiskArray) = haschunks(parent(A))
eachchunk(A::PaddedDiskArray) = A.chunks

readblock!(A::PaddedDiskArray, data, I::AbstractRange...) =
    _readblock_padded(A, data, I...)
writeblock!(A::PaddedDiskArray, data, I...) = 
    throw(ArgumentError("Cannot write to a PaddedDiskArray"))

function _readblock_padded(A, data, I::AbstractRange...)
    data .= A.fill
    Ipadded = map(I, A.padding) do i, (low, high)
        i .- low
    end
    fs = map(axes(parent(A)), Ipadded) do a, ip
        searchsortedfirst(ip, first(a))
    end
    ls = map(axes(parent(A)), Ipadded) do a, ip
        searchsortedlast(ip, last(a))
    end
    return if all(map(<=, fs, ls))
        Idata = map(:, fs, ls)
        Iparent = map(getindex, Ipadded, Idata)
        data[Idata...] .= parent(A)[Iparent...]
    else
        # No overlap, don't read
        data
    end
end

"""

    pad(A, padding; fill=zero(eltype(A)))

Pad any `AbstractArray` with fill values, updating chunk patterns.

# Arguments

- `A`: The parent disk array.
- `padding`: A tuple of `Int` lower and upper padding tuples, one for each dimension.

# Keywords

- `fill=zero(eltype(A))`: The value to pad the array with.
"""
pad(A::AbstractArray{<:Any,N}, padding::NTuple{N,Tuple{Int,Int}}; kw...) where N = 
    PaddedDiskArray(A, padding; kw...)