"""
    DiskIndex

    DiskIndex(
        output_size::NTuple{N,<:Integer},
        temparray_size::NTuple{M,<:Integer}, 
        output_indices::Tuple,
        temparray_indices::Tuple,
        data_indices::Tuple
    )
    DiskIndex(a::AbsractArray, i)

An object encoding indexing into a chunked disk array,
and to memory-backed input/output buffers.

# Arguments and fields

- `output_size` size of the output array
- `temparray_size` size of the temp array passed to `readblock`
- `output_indices` indices for copying into the output array
- `temparray_indices` indices for reading from temp array
- `data_indices` indices for reading from data array
"""
struct DiskIndex{N,M,A<:Tuple,B<:Tuple,C<:Tuple}
    output_size::NTuple{N,Int}
    temparray_size::NTuple{M,Int}
    output_indices::A
    temparray_indices::B
    data_indices::C
end
function DiskIndex(
    output_size::Tuple{Vararg{Integer}},
    temparray_size::Tuple{Vararg{Integer}},
    output_indices::Tuple,
    temparray_indices::Tuple,
    data_indices::Tuple
)
    output_size_int = map(Int, output_size)
    temparray_size_int = map(Int, temparray_size)
    DiskIndex(output_size_int, temparray_size_int, output_indices, temparray_indices, data_indices)
end
DiskIndex(a, i) = DiskIndex(a, i, batchstrategy(a))
DiskIndex(a, i, batch_strategy) =
    _resolve_indices(eachchunk(a).chunks, i, DiskIndex((), (), (), (), ()), batch_strategy)
DiskIndex(a::AbstractVector, i::Tuple{AbstractVector{<:Integer}}, batch_strategy) =
    _resolve_indices(eachchunk(a).chunks, i, DiskIndex((), (), (), (), ()), batch_strategy)
DiskIndex(a, ::Tuple{Colon}, _) =
    DiskIndex((length(a),), size(a), (Colon(),), (Colon(),), map(s -> 1:s, size(a)))
DiskIndex(a, i::Tuple{<:CartesianIndex}, batch_strategy=NoBatch()) =
    DiskIndex(a, only(i).I, batch_strategy)
DiskIndex(a, i::Tuple{<:AbstractVector{<:Integer}}, batchstrategy) =
    DiskIndex(a, (view(CartesianIndices(a), only(i)),), batchstrategy)

function _resolve_indices(chunks, i, indices_pre::DiskIndex, strategy::BatchStrategy)
    inow = first(i)
    indices_new, chunksrem = process_index(inow, chunks, strategy)
    _resolve_indices(chunksrem, tail(i), merge_index(indices_pre, indices_new), strategy)
end
# Some (pretty stupid) hacks to get around Base recursion limiting https://github.com/JuliaLang/julia/pull/48059
# TODO: We can remove these if Base sorts this out.
# This makes 3 arg type stable
function _resolve_indices(chunks::Tuple{<:Any}, i::Tuple{<:Any}, indices_pre::DiskIndex, strategy::BatchStrategy)
    inow = first(i)
    indices_new, chunksrem = process_index(inow, chunks, strategy)
    return merge_index(indices_pre, indices_new)
end
# This makes 4 arg type stable
function _resolve_indices(chunks::Tuple{<:Any,<:Any}, i::Tuple{<:Any,<:Any}, indices_pre::DiskIndex, strategy::BatchStrategy)
    inow = first(i)
    indices_new, chunksrem = process_index(inow, chunks, strategy)
    return _resolve_indices(chunksrem, tail(i), merge_index(indices_pre, indices_new), strategy)
end
# This makes 5 arg type stable
function _resolve_indices(chunks::Tuple{<:Any,<:Any,<:Any}, i::Tuple{<:Any,<:Any,<:Any}, indices_pre::DiskIndex, strategy::BatchStrategy)
    inow = first(i)
    indices_new, chunksrem = process_index(inow, chunks, strategy)
    return _resolve_indices(chunksrem, tail(i), merge_index(indices_pre, indices_new), strategy)
end
# This makes 6 arg type stable
function _resolve_indices(chunks::Tuple{<:Any,<:Any,<:Any,<:Any}, i::Tuple{<:Any,<:Any,<:Any,<:Any}, indices_pre::DiskIndex, strategy::BatchStrategy)
    inow = first(i)
    indices_new, chunksrem = process_index(inow, chunks, strategy)
    return _resolve_indices(chunksrem, tail(i), merge_index(indices_pre, indices_new), strategy)
end
# Splat out CartesianIndex as regular indices
function _resolve_indices(
    chunks::Tuple, i::Tuple{<:CartesianIndex}, indices_pre::DiskIndex, strategy::BatchStrategy
)
    _resolve_indices(chunks, (Tuple(i[1])..., tail(i)...), indices_pre, strategy)
end
# This method is needed to resolve ambiguity
function _resolve_indices(
    chunks::Tuple{<:Any}, i::Tuple{<:CartesianIndex}, indices_pre::DiskIndex, strategy::BatchStrategy
)
    _resolve_indices(chunks, (Tuple(i[1])..., tail(i)...), indices_pre, strategy)
end
_resolve_indices(::Tuple{}, ::Tuple{}, indices::DiskIndex, strategy::BatchStrategy) = indices
# No dimension left in array, only singular indices allowed
function _resolve_indices(::Tuple{}, i, indices_pre::DiskIndex, strategy::BatchStrategy)
    inow = first(i)
    (length(inow) == 1 && only(inow) == 1) || throw(ArgumentError("Trailing indices must be 1"))
    indices_new = DiskIndex(size(inow), (), size(inow), (), ())
    indices = merge_index(indices_pre, indices_new)
    _resolve_indices((), tail(i), indices, strategy)
end
# Splat out CartesianIndex as regular trailing indices
function _resolve_indices(
    ::Tuple{}, i::Tuple{<:CartesianIndex}, indices_pre::DiskIndex, strategy::BatchStrategy
)
    _resolve_indices((), (Tuple(i[1])..., tail(i)...), indices_pre, strategy)
end
# Still dimensions left, but no indices available
function _resolve_indices(chunks, ::Tuple{}, indices_pre::DiskIndex, strategy::BatchStrategy)
    chunksnow = first(chunks)
    checktrailing(arraysize_from_chunksize(chunksnow))
    indices_new = add_dimension_index(strategy)
    indices = merge_index(indices_pre, indices_new)
    _resolve_indices(tail(chunks), (), indices, strategy)
end

checktrailing(i) = i == 1 || throw(ArgumentError("Indices can only be omitted for trailing singleton dimensions"))

add_dimension_index(::NoBatch) = DiskIndex((), (1,), (), (1,), (1:1,))
add_dimension_index(::Union{ChunkRead,SubRanges}) = DiskIndex((), (1,), ([()],), ([(1,)],), ([(1:1,)],))

"""
    merge_index(a::DiskIndex, b::DiskIndex)

Merge two `DiskIndex` into a single index accross more dimensions.
"""
@inline function merge_index(a::DiskIndex, b::DiskIndex)
    DiskIndex(
        (a.output_size..., b.output_size...),
        (a.temparray_size..., b.temparray_size...),
        (a.output_indices..., b.output_indices...),
        (a.temparray_indices..., b.temparray_indices...),
        (a.data_indices..., b.data_indices...),
    )
end

"""
    process_index(i, chunks, batchstrategy)

Calculate indices for `i` the first chunk/s in `chunks`

Returns a [`DiskIndex`](@ref), and the remaining chunks.
"""
process_index(i, chunks, ::NoBatch) = process_index(i, chunks)
function process_index(i::CartesianIndex{N}, chunks::Tuple, ::NoBatch) where {N}
    _, chunksrem = splitchunks(i, chunks)
    di = DiskIndex((), map(one, i.I), (), (1,), map(i -> i:i, i.I))

    return di, chunksrem
end
process_index(inow::Integer, chunks) = 
    DiskIndex((), (1,), (), (1,), (inow:inow,)), tail(chunks)
function process_index(::Colon, chunks)
    s = arraysize_from_chunksize(first(chunks))
    di = DiskIndex((s,), (s,), (Colon(),), (Colon(),), (1:s,),)
    return di, tail(chunks)
end
function process_index(i::AbstractUnitRange{<:Integer}, chunks, ::NoBatch)
    di = DiskIndex((length(i),), (length(i),), (Colon(),), (Colon(),), (i,))
    return di::DiskIndex, tail(chunks)::Tuple
end
function process_index(i::AbstractArray{<:Integer}, chunks, ::NoBatch)
    indmin, indmax = isempty(i) ? (1, 0) : extrema(i)

    output_size = size(i)
    temparray_size = ((indmax - indmin + 1),)
    output_indices = map(_ -> Colon(), size(i))
    temparray_indices = ((i .- (indmin - 1)),)
    data_indices = (indmin:indmax,)
    di = DiskIndex(output_size, temparray_size, output_indices, temparray_indices, data_indices)

    return di, tail(chunks)
end
function process_index(i::AbstractArray{Bool,N}, chunks, ::NoBatch) where {N}
    chunksnow, chunksrem = splitchunks(i, chunks)
    s = arraysize_from_chunksize.(chunksnow)
    cindmin, cindmax = extrema(view(CartesianIndices(s), i))
    indmin, indmax = cindmin.I, cindmax.I

    output_size = (sum(i),)
    temparray_size = map((max, min) -> max - min + 1, indmax, indmin)
    output_indices = (Colon(),)
    temparray_indices = (view(i, map(range, indmin, indmax)...),)
    data_indices = map(range, indmin, indmax)
    di = DiskIndex(output_size, temparray_size, output_indices, temparray_indices, data_indices)

    return di, chunksrem
end
function process_index(i::AbstractArray{<:CartesianIndex{N}}, chunks, ::NoBatch) where {N}
    chunksnow, chunksrem = splitchunks(i, chunks)
    s = arraysize_from_chunksize.(chunksnow)
    v = view(CartesianIndices(s), i)
    cindmin, cindmax = if isempty(v)
        oneunit(CartesianIndex{N}), zero(CartesianIndex{N})
    else
        extrema(v)
    end
    indmin, indmax = cindmin.I, cindmax.I

    output_size = size(i)
    temparray_size = map((max, min) -> max - min + 1, indmax, indmin)
    temparray_offset = cindmin - oneunit(cindmin)
    temparray_indices = (i .- (CartesianIndex(temparray_offset),),)
    output_indices = map(_ -> Colon(), size(i))
    data_indices = map(range, indmin, indmax)
    di = DiskIndex(output_size, temparray_size, output_indices, temparray_indices, data_indices)

    return di, chunksrem
end
function process_index(i::CartesianIndices{N}, chunks, ::NoBatch) where {N}
    _, chunksrem = splitchunks(i, chunks)

    output_size = map(length, i.indices)  
    temparray_size = map(length, i.indices)
    output_indices = temparray_indices = map(_ -> Colon(), i.indices)
    data_indices = i.indices
    di = DiskIndex(output_size, temparray_size, output_indices, temparray_indices, data_indices)

    return di, chunksrem
end

"""
    splitchunks(i, chunks)

Split chunks into a 2-tuple based on i, so that the first group
match i and the second match the remaining indices.

The dimensionality of `i` will determine the number of chunks
returned in the first group.
"""
splitchunks(i::AbstractArray{<:CartesianIndex}, chunks) =
    splitchunks(oneunit(eltype(i)).I, (), chunks)
splitchunks(i::AbstractArray{Bool}, chunks) = splitchunks(size(i), (), chunks)
splitchunks(i::CartesianIndices, chunks) = splitchunks(i.indices, (), chunks)
splitchunks(i::CartesianIndex, chunks) = splitchunks(i.I, (), chunks)
splitchunks(_, chunks) = (first(chunks),), Base.tail(chunks)
splitchunks(si, chunksnow, chunksrem) =
    splitchunks(Base.tail(si), (chunksnow..., first(chunksrem)), Base.tail(chunksrem))
function splitchunks(si,chunksnow, ::Tuple{})
    only(first(si)) == 1 || throw(ArgumentError("Trailing indices must be 1"))
    splitchunks(Base.tail(si), chunksnow, ())
end
splitchunks(::Tuple{}, chunksnow, chunksrem) = (chunksnow, chunksrem)
splitchunks(::Tuple{}, chunksnow, chunksrem::Tuple{}) = (chunksnow, chunksrem)

"""
    output_aliasing(di::DiskIndex, ndims_dest, ndims_source)

Determines wether output and temp array can:

a) be identical, returning `:identical`
b) share memory through reshape, returning `:reshapeoutput` 
c) need to be allocated individually, returning `:noalign`
"""
function output_aliasing(di::DiskIndex, ndims_dest, ndims_source)
    if all(i -> i isa Union{Int,AbstractUnitRange,Colon}, di.temparray_indices) &&
       all(i -> i isa Union{Int,AbstractUnitRange,Colon}, di.output_indices)
        if di.output_size == di.temparray_size && ndims_dest == ndims_source
            return :identical
        else
            return :reshapeoutput
        end
    else
        return :noalign
    end
end

