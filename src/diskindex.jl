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
    output_size::NTuple{N,<:Integer},
    temparray_size::NTuple{M,<:Integer}, 
    output_indices::Tuple,
    temparray_indices::Tuple,
    data_indices::Tuple
) where {N,M}
    DiskIndex(Int.(output_size), Int.(temparray_size), output_indices, temparray_indices, data_indices)
end
DiskIndex(a, i) = DiskIndex(a, i, batchstrategy(a))
DiskIndex(a, i, batch_strategy) = 
    _resolve_indices(eachchunk(a).chunks, i, DiskIndex((),(),(),(),()), batch_strategy)
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
    _resolve_indices(chunksrem, Base.tail(i), merge_index(indices_pre, indices_new), strategy)
end
_resolve_indices(::Tuple{}, ::Tuple{}, indices::DiskIndex, strategy::BatchStrategy) = indices
# No dimension left in array, only singular indices allowed
function _resolve_indices(::Tuple{}, i, indices_pre::DiskIndex, strategy::BatchStrategy)
    inow = first(i)
    (length(inow) == 1 && only(inow) == 1) || throw(ArgumentError("Trailing indices must be 1"))
    indices_new = DiskIndex(size(inow),(),size(inow),(),())
    indices = merge_index(indices_pre,indices_new)
    _resolve_indices((), Base.tail(i), indices, strategy)
end
# Still dimensions left, but no indices available
function _resolve_indices(chunks, ::Tuple{}, indices_pre::DiskIndex, strategy::BatchStrategy) 
    chunksnow = first(chunks)
    arraysize_from_chunksize(chunksnow) == 1 || throw(ArgumentError("Indices can only be omitted for trailing singleton dimensions"))
    indices_new = add_dimension_index(strategy)
    indices = merge_index(indices_pre,indices_new)
    _resolve_indices(Base.tail(chunks), (), indices, strategy)
end

add_dimension_index(::NoBatch) = DiskIndex((),(1,),(),(1,),(1:1,))
add_dimension_index(::Union{ChunkRead,SubRanges}) = DiskIndex((),(1,),([()],),([(1,)],),([(1:1,)],))

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
process_index(inow::Integer, chunks) = DiskIndex((), (1,), (), (1,), (inow:inow,)), Base.tail(chunks)
function process_index(::Colon, chunks)
    s = arraysize_from_chunksize(first(chunks))
    DiskIndex((s,), (s,), (Colon(),), (Colon(),), (1:s,),), Base.tail(chunks)
end
function process_index(i::AbstractUnitRange{<:Integer}, chunks, ::NoBatch)
    DiskIndex((length(i),), (length(i),), (Colon(),), (Colon(),), (i,)), Base.tail(chunks)
end
function process_index(i::AbstractArray{<:Integer}, chunks, ::NoBatch)
    indmin, indmax = isempty(i) ? (1,0) : extrema(i)
    di = DiskIndex(size(i), ((indmax - indmin + 1),), map(_->Colon(),size(i)), ((i .- (indmin - 1)),), (indmin:indmax,))
    return di, Base.tail(chunks)
end
function process_index(i::AbstractArray{Bool,N}, chunks, ::NoBatch) where {N}
    chunksnow, chunksrem = splitchunks(i, chunks)
    s = arraysize_from_chunksize.(chunksnow)
    cindmin, cindmax = extrema(view(CartesianIndices(s), i))
    indmin, indmax = cindmin.I, cindmax.I
    tempsize = indmax .- indmin .+ 1
    tempinds = view(i, range.(indmin, indmax)...)
    di = DiskIndex((sum(i),), tempsize, (Colon(),), (tempinds,), range.(indmin, indmax))
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
    tempsize = indmax .- indmin .+ 1
    tempoffset = cindmin - oneunit(cindmin)
    tempinds = i .- (CartesianIndex(tempoffset),)
    outinds = map(_->Colon(),size(i))
    di = DiskIndex(size(i), tempsize, outinds, (tempinds,), range.(indmin, indmax))
    return di, chunksrem
end
function process_index(i::CartesianIndices{N}, chunks, ::NoBatch) where {N}
    _, chunksrem = splitchunks(i, chunks)
    cols = map(_ -> Colon(), i.indices)
    di = DiskIndex(length.(i.indices), length.(i.indices), cols, cols, i.indices)
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
splitchunks(i::CartesianIndex, chunks) = splitchunks(i.I,(),chunks)
splitchunks(_, chunks) = (first(chunks),), Base.tail(chunks)
splitchunks(si, chunksnow, chunksrem) = 
    splitchunks(Base.tail(si), (chunksnow..., first(chunksrem)), Base.tail(chunksrem))
splitchunks(::Tuple{}, chunksnow, chunksrem) = (chunksnow, chunksrem)

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

