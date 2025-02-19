"""
    ChunkVector <: AbstractVector{UnitRange}

Supertype for lazy vectors of `UnitRange`.

[`RegularChunks`](@ref) and [`IrregularChunks`](@ref) 
are the implementations.
"""
abstract type ChunkVector <: AbstractVector{UnitRange} end

findchunk(a::ChunkVector, i::AbstractUnitRange) =
    (findchunk(a, first(i))::Int):(findchunk(a, last(i))::Int)
findchunk(a::ChunkVector, ::Colon) = 1:length(a)

# Identify chunks from indices which may be discontinuous.
subsetchunks(chunks::ChunkVector, subs::BitVector) = subsetchunks(chunks, findall(subs))
subsetchunks(chunks::ChunkVector, subsets) = subsetchunks_fallback(chunks, subsets)

function subsetchunks_fallback(chunks::ChunkVector, subsets)
    # This is a fallback method that should work for Regular and Irregular chunks.
    # Assuming the desired subset is sorted, we simply compute the chunk for every element 
    # in subs and collect everything together again in either a Regular or IrregularChunk
    if isempty(subsets)
        return RegularChunks(1, 0, 0)
    end
    rev = if issorted(subsets)
        false
    elseif issorted(subsets; rev=true)
        true
    else
        runlength = chunk_runlength(chunks, subsets)
        return chunktype_from_chunksizes(runlength)
    end
    chunksizes = zeros(Int, length(chunks))
    for i in subsets
        chunksizes[findchunk(chunks, i)] += 1
    end
    # Find first and last chunk where elements are extracted
    i1 = findfirst(!iszero, chunksizes)
    i2 = findlast(!iszero, chunksizes)
    if isnothing(i1) || isnothing(i2)
        error("Should not be reached. Non-zero indices not found")
    else
        newchunksizes = chunksizes[i1:i2]
        if rev
            reverse!(newchunksizes)
        end
        return chunktype_from_chunksizes(newchunksizes)
    end
end

"""
    RegularChunks <: ChunkArray

Defines chunking along a dimension where the chunks have constant size and a potential
offset for the first chunk. The last chunk is truncated to fit the array size. 
"""
struct RegularChunks <: ChunkVector
    chunksize::Int
    offset::Int
    arraysize::Int
    function RegularChunks(chunksize::Integer, offset::Integer, arraysize::Integer)
        chunksize > 0 || throw(ArgumentError("Chunk sizes must be strictly positive"))
        -1 < offset < chunksize || throw(ArgumentError("Offsets must be positive and smaller than the chunk size, got offset: $offset and chunk size: $chunksize"))
        arraysize >= 0 || throw(ArgumentError("Negative axis lengths are not allowed"))
        new(Int(chunksize), Int(offset), Int(arraysize))
    end
end

function Base.show(io::IO, chunks::RegularChunks)
    (; chunksize, offset, arraysize) = chunks
    Base.print(io, "RegularChunks($chunksize, $offset, $arraysize)")
end

# Base methods

Base.@propagate_inbounds function Base.getindex(r::RegularChunks, i::Int)
    @boundscheck checkbounds(r, i)
    return max((i - 1) * r.chunksize + 1 - r.offset, 1):min(i * r.chunksize - r.offset, r.arraysize)
end
Base.size(r::RegularChunks, _) = div(r.arraysize + r.offset - 1, r.chunksize) + 1
Base.size(r::RegularChunks) = (size(r, 1),)
function Base.:(==)(r1::RegularChunks, r2::RegularChunks)
    # The axis sizes must always match
    r1.arraysize == r2.arraysize || return false
    # The number of chunks must also match
    nchunks = length(r1)
    nchunks == length(r2) || return false
    # But after that we need to take the number of chunks into account
    if nchunks > 2 
        # For longer RegularChunks the offsets and chunk sizes 
        # must match for the chunks to be the same. 
        # So we compare them directly rather than iterating all of the ranges
        return r1.chunksize == r2.chunksize && r1.offset == r2.offset
    elseif nchunks == 2
        # Smaller RegularChunks can match with different chunk sizes and offsets
        # So we compare the ranges
        return first(r1) == first(r2) && last(r1) == last(r2)
    elseif nchunks == 1
        return first(r1) == first(r2)
    else
        return true
    end
end

# DiskArrays interface

function subsetchunks(chunks::RegularChunks, subsets::AbstractUnitRange)
    newsize = length(subsets)
    newoffset = mod(first(subsets) - 1 + chunks.offset, chunks.chunksize)
    chunks = RegularChunks(chunks.chunksize, newoffset, newsize)
    # In case the new chunk is trivial and has length 1, we shorten the chunk size
    if length(chunks) == 1
        chunks = RegularChunks(max(newsize, 1), 0, newsize)
    end
    return chunks
end
# Need to handle step sizes for AbstractRange
function subsetchunks(chunks::RegularChunks, subsets::AbstractRange)
    # Check if the chunksize is divisible by the step size of the subsets range
    if rem(chunks.chunksize, step(subsets)) == 0
        # In which cas the chunk size is divided by the step size
        newchunksize = chunks.chunksize ÷ abs(step(subsets))
        if step(subsets) > 0
            newoffset = mod(first(subsets) - 1 + chunks.offset, chunks.chunksize) ÷ step(subsets)
            return RegularChunks(newchunksize, newoffset, length(subsets))
        elseif step(subsets) < 0
            chunks2 = subsetchunks(chunks, last(subsets):first(subsets))::ChunkVector
            newoffset = (chunks.chunksize - length(last(chunks2))) ÷ (-step(subsets))
            return RegularChunks(newchunksize, newoffset, length(subsets))
        end
    else
        return subsetchunks_fallback(chunks, subsets)
    end
end

findchunk(chunks::RegularChunks, i::Int) =
    div(i + chunks.offset - 1, chunks.chunksize) + 1

approx_chunksize(r::RegularChunks) = r.chunksize
grid_offset(r::RegularChunks) = r.offset
max_chunksize(r::RegularChunks) = r.chunksize

"""
    IrregularChunks <: ChunkVector

Defines chunks along a dimension where chunk sizes are not constant but arbitrary
"""
struct IrregularChunks <: ChunkVector
    offsets::Vector{Int}
    function IrregularChunks(offsets::Vector{Int})
        first(offsets) == 0 ||
            throw(ArgumentError("First Offset of an Irregularchunk must be 0"))
        all(i -> offsets[i] < offsets[i+1], 1:(length(offsets)-1)) ||
            throw(ArgumentError("Offsets of an Irregularchunk must be strictly ordered"))
        return new(offsets)
    end
end
"""
    IrregularChunks(; chunksizes)

Returns an IrregularChunks object for the given list of chunk sizes
"""
function IrregularChunks(; chunksizes)
    offsets = pushfirst!(cumsum(chunksizes), 0)
    # push!(offs, last(offs)+1)
    return IrregularChunks(offsets)
end

Base.@propagate_inbounds function Base.getindex(chunks::IrregularChunks, i::Int)
    @boundscheck checkbounds(chunks, i)
    return (chunks.offsets[i]+1):chunks.offsets[i+1]
end
  
Base.size(chunks::IrregularChunks) = (length(chunks.offsets) - 1,)
Base.:(==)(c1::IrregularChunks, c2::IrregularChunks) =
    c1 === c2 || c1.offsets == c2.offsets
Base.show(io::IO, chunks::IrregularChunks) =
    Base.print(io, "IrregularChunks($(chunks.offsets))")

function subsetchunks(chunks::IrregularChunks, subsets::UnitRange)
    if isempty(subsets)
        return IrregularChunks([0])
    end
    c1 = findchunk(chunks, first(subsets))
    c2 = findchunk(chunks, last(subsets))
    newoffsets = chunks.offsets[c1:(c2+1)]
    firstoffset = first(subsets) - chunks.offsets[c1] - 1
    newoffsets[end] = last(subsets)
    newoffsets[2:end] .= newoffsets[2:end] .- firstoffset
    newoffsets .= newoffsets .- first(newoffsets)
    return IrregularChunks(newoffsets)
end
findchunk(chunks::IrregularChunks, i::Int) =
    searchsortedfirst(chunks.offsets, i) - 1
approx_chunksize(chunks::IrregularChunks) =
    round(Int, sum(diff(chunks.offsets)) / (length(chunks.offsets) - 1))
grid_offset(chunks::IrregularChunks) = 0
max_chunksize(chunks::IrregularChunks) = maximum(diff(chunks.offsets))

"""
    GridChunks

Multi-dimensional chunk specification, that holds a chunk pattern 
for each axis of an array. 

These are usually `RegularChunks` or `IrregularChunks`.
"""
struct GridChunks{N,C<:Tuple{Vararg{ChunkVector,N}}} <:
       AbstractArray{NTuple{N,UnitRange{Int64}},N}
    chunks::C
end
GridChunks(chunks::ChunkVector...) = GridChunks(chunks)
GridChunks(a, chunksize; kw...) = GridChunks(size(a), chunksize; kw...)
function GridChunks(sizes::Tuple, chunksizes::Tuple; offset=(_ -> 0).(sizes))
    chunks = map(RegularChunks, chunksizes, offset, sizes)
    return GridChunks(chunks)
end
function Base.show(io::IO, gridchunks::GridChunks)
    println(io, "GridChunks(")
    map(gridchunks.chunks) do chunks
        println("    ")
        show(io, chunks)
        print(",")
    end
    println(io, ")")
end

# Base methods

Base.@propagate_inbounds function Base.getindex(gc::GridChunks{N}, i::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(gc, i...)
    return map(getindex, gc.chunks, i)
end
Base.size(gc::GridChunks) = map(length, gc.chunks)
Base.:(==)(gc1::GridChunks, gc2::GridChunks) = gc1.chunks == gc2.chunks

# Utils

# Computes the run length of which chunk is accessed how many times consecutively.
function chunk_runlength(chunks::ChunkVector, vec)
    out = Int[]
    currentchunk = -1
    for i in vec
        nextchunk = findchunk(chunks, i)
        if nextchunk == currentchunk
            out[end] += 1
        else
            push!(out, 1)
            currentchunk = nextchunk
        end
    end
    return out
end

# Utility function that constructs either a `RegularChunks` or an
# `IrregularChunks` object based on a vector of chunk sizes given as worted Integers. Wherever
# possible it will try to create a regular chunks object.  
function chunktype_from_chunksizes(chunksizes::AbstractVector)
    if length(chunksizes) == 1
        # only a single chunk is affected
        return RegularChunks(max(chunksizes[1], 1), 0, chunksizes[1])
    elseif length(chunksizes) == 2
        # Two affected chunks
        chunksize = max(chunksizes[1], chunksizes[2])
        return RegularChunks(chunksize, chunksize - chunksizes[1], sum(chunksizes))
    elseif all(==(chunksizes[2]), view(chunksizes, (2):(length(chunksizes)-1))) &&
           chunksizes[end] <= chunksizes[2] &&
           chunksizes[1] <= chunksizes[2]
        # All chunks have the same size, only first and last chunk can be shorter
        chunksize = chunksizes[2]
        return RegularChunks(chunksize, chunksize - chunksizes[1], sum(chunksizes))
    else
        # Chunks are Irregular
        return IrregularChunks(; chunksizes=filter(!iszero, chunksizes))
    end
end

"""
    arraysize_from_chunksize(g::ChunkVector)

Returns the size of the dimension represented by a chunk object. 
"""
arraysize_from_chunksize(chunks::RegularChunks) = chunks.arraysize
arraysize_from_chunksize(chunks::IrregularChunks) = last(chunks.offsets)

# DiskArrays interface

"""
    approx_chunksize(g::GridChunks)

Returns the aproximate chunk size of the grid. 

For the dimension with regular chunks, this will be the exact chunk size
while for dimensions with irregular chunks this is the average chunks size. 

Useful for downstream applications that want to
distribute computations and want to know about chunk sizes. 
"""
approx_chunksize(g::GridChunks) = map(approx_chunksize, g.chunks)

"""
    grid_offset(g::GridChunks)

Returns the offset of the grid for the first chunks. 

Expect this value to be non-zero for views into regular-gridded arrays. 

Useful for downstream applications that want to 
distribute computations and want to know about chunk sizes. 
"""
grid_offset(g::GridChunks) = map(grid_offset, g.chunks)

"""
    max_chunksize(g::GridChunks)

Returns the maximum chunk size of an array for each dimension. 

Useful for pre-allocating arrays to make sure they can hold a chunk of data. 
"""
max_chunksize(g::GridChunks) = map(max_chunksize, g.chunks)

# Define the approx default maximum chunk size (in MB)
"The target chunk size for processing for unchunked arrays in MB, defaults to 100MB"
const default_chunk_size = Ref(100)

"""
A fallback element size for arrays to determine a where elements have unknown
size like strings. Defaults to 100MB
"""
const fallback_element_size = Ref(100)


"""
    ChunkedTrait{S}

Traits for disk array chunking. 

[`Chunked`](@ref) or [`Unchunked`](@ref).

Always hold a [`BatchStrategy`](@ref) trait.
"""
abstract type ChunkedTrait{BS} end

batchstrategy(trait::ChunkedTrait) = trait.batchstrategy

"""
    Chunked{<:BatchStrategy}

A trait that specifies an Array has a chunked read pattern.
"""
struct Chunked{BS} <: ChunkedTrait{BS}
    batchstrategy::BS
end
Chunked() = Chunked(ChunkRead())

"""
    Unchunked{<:BatchStrategy}

A trait that specifies an Array does not have a chunked read pattern,
and random access indexing is relatively performant.
"""
struct Unchunked{BS} <: ChunkedTrait{BS}
    batchstrategy::BS
end
Unchunked() = Unchunked(SubRanges())


"""
   ChunkIndexType

Triats for [`ChunkIndex`](@ref).

`OffsetChunks()` or `OneBasedChunks()`.
"""
abstract type ChunkIndexType end

struct OffsetChunks <: ChunkIndexType end
struct OneBasedChunks <: ChunkIndexType end

wrapchunk(x, inds) = OffsetArray(x, inds...)

"""
    ChunkIndex{N}

This can be used in indexing operations when one wants to 
extract a full data chunk from a DiskArray. 

Useful for iterating over chunks of data. 

`d[ChunkIndex(1, 1)]` will extract the first chunk of a 2D-DiskArray
"""
struct ChunkIndex{N,O<:ChunkIndexType}
    I::CartesianIndex{N}
    chunktype::O
end
ChunkIndex(i::CartesianIndex; offset=false) =
    ChunkIndex(i, offset ? OffsetChunks() : OneBasedChunks())
ChunkIndex(i::Integer...; kw...) = ChunkIndex(CartesianIndex(i); kw...)

"Removes the offset from a ChunkIndex"
nooffset(i::ChunkIndex) = ChunkIndex(i.I, OneBasedChunks())

"""
    ChunkIndices{N}

Represents an iterator of `ChunkIndex` objects.
"""
struct ChunkIndices{N,RT<:Tuple{Vararg{Any,N}},O} <: AbstractArray{ChunkIndex{N},N}
    I::RT
    chunktype::O
end

Base.size(i::ChunkIndices) = length.(i.I)
Base.getindex(A::ChunkIndices{N}, I::Vararg{Int,N}) where {N} =
    ChunkIndex(CartesianIndex(getindex.(A.I, I)), A.chunktype)
Base.eltype(::Type{<:ChunkIndices{N}}) where {N} = ChunkIndex{N}

"""
    element_size(a::AbstractArray)

Returns the approximate size of an element of a in bytes. This falls back to calling `sizeof` on 
the element type or to the value stored in `DiskArrays.fallback_element_size`. Methods can be added for 
custom containers. 
"""
function element_size(a::AbstractArray)
    if isbitstype(eltype(a))
        return sizeof(eltype(a))
    elseif isbitstype(Base.nonmissingtype(eltype(a)))
        return sizeof(Base.nonmissingtype(eltype(a)))
    else
        return fallback_element_size[]
    end
end

"""
    estimate_chunksize(a::AbstractArray)

Estimate a suitable chunk pattern for an `AbstractArray` without chunks.
"""
estimate_chunksize(a::AbstractArray) = estimate_chunksize(size(a), element_size(a))
function estimate_chunksize(size, elsize)
    ii = searchsortedfirst(cumprod(collect(size)), default_chunk_size[] * 1e6 / elsize)
    chunksize = ntuple(length(size)) do idim
        if idim < ii
            return size[idim]
        elseif idim > ii
            return 1
        else
            sizebefore = idim == 1 ? 1 : prod(size[1:(idim-1)])
            return floor(Int, default_chunk_size[] * 1e6 / elsize / sizebefore)
        end
    end
    chunksizes = clamp.(chunksize, 1, size)
    return GridChunks(size, chunksize)
end
