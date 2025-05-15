"""
    MultiReadArray <: AbstractArray

An array too that holds indices for multiple block reads.
"""
struct MultiReadArray{T,N,A<:Tuple} <: AbstractArray{T,N}
    a::A
end
function MultiReadArray(a)
    MultiReadArray{Any,length(a),typeof(a)}(a)
end

Base.size(a::MultiReadArray) = _mapflatten(length, a.a)
Base.IndexStyle(::Type{<:MultiReadArray}) = IndexCartesian()
Base.eachindex(a::MultiReadArray) = CartesianIndices(size(a))
Base.getindex(a::MultiReadArray{<:Any,N}, I::Vararg{Int,N}) where {N} =
    map(getindex, a.a, I) |> _flatten1

_mapflatten(f, x) = foldl((x, y) -> (x..., f(y)), x; init=())
_flatten1(a) = _flatten(first(a), Base.tail(a))
_flatten(r, a) = _flatten((r..., first(a)...), Base.tail(a))
_flatten(r, ::Tuple{}) = r

"""
    getindex_disk(a::AbstractArray, i...)

Internal `getindex` for disk arrays.

Converts indices to ranges and calls `DiskArrays.readblock!`
"""
function getindex_disk(a::AbstractArray, i::Union{Integer,CartesianIndex}...)
    checkscalar(a, i)
    checkbounds(a, i...)
    # Use a 1 x 1 block
    outputarray = Array{eltype(a)}(undef, map(_ -> 1, size(a))...)
    i = Base.to_indices(a, i)
    # Convert indices to length 1 ranges
    j = map(1:ndims(a)) do d
        d <= length(i) ? (i[d]:i[d]) : 1:1
    end
    # Read the block
    readblock!(a, outputarray, j...)
    # Return the only value
    return only(outputarray)
end
function getindex_disk(a::AbstractArray, i::Integer)
    checkscalar(a, i)
    checkbounds(a, i)
    # Use a 1 x 1 block
    outputarray = Array{eltype(a)}(undef, map(_ -> 1, size(a))...)
    # Convert linear index to length 1 cartesian ranges
    j = map(k -> k:k, CartesianIndices(a)[i].I)
    # Read the block
    readblock!(a, outputarray, j...)
    # Return the only value
    return only(outputarray)
end
getindex_disk(a::AbstractArray, i...) = getindex_disk!(nothing, a, i...)
getindex_disk(a::AbstractArray, i::ChunkIndex{<:Any,OneBasedChunks}) =
    a[eachchunk(a)[i.I]...]
getindex_disk(a::AbstractArray, i::ChunkIndex{<:Any,OffsetChunks}) =
    wrapchunk(a[nooffset(i)], eachchunk(a)[i.I])

function getindex_disk!(out::Union{Nothing,AbstractArray}, a::AbstractArray, i...)
    # Check if we can write once or need to use multiple batches
    if need_batch(a, i)
        getindex_disk_batch!(out, a, i)
    else
        getindex_disk_nobatch!(out, a, i)
    end
end

function getindex_disk_batch!(out::Union{Nothing,AbstractArray}, a::AbstractArray, i)
    # Convert indices to DiskIndex
    indices = DiskIndex(a, i)
    # Generate batched read indices
    # TODO: this could be a single object wrapping DiskIndices
    moutput_indices = MultiReadArray(indices.output_indices)
    mtemparray_indices = MultiReadArray(indices.temparray_indices)
    mdata_indicess = MultiReadArray(indices.data_indices)
    # Generate the output Array
    outputarray = create_outputarray(out, a, indices.output_size)
    # And a temp array to pass to `readblock!`
    temparray = Array{eltype(a)}(undef, indices.temparray_size...)
    # Iterate over indices for each batch read
    for ii in eachindex(moutput_indices)
        data_indices = mdata_indicess[ii]
        output_indices = moutput_indices[ii]
        temparray_indices = mtemparray_indices[ii]
        vtemparray = maybeshrink(temparray, data_indices)
        readblock!(a, vtemparray, data_indices...)
        transfer_results_read!(outputarray, temparray, output_indices, temparray_indices)
    end
    return outputarray
end

function getindex_disk_nobatch!(out::Union{Nothing,AbstractArray}, a::AbstractArray, i)
    strategy = NoBatch(allow_steprange(a) ? CanStepRange() : NoStepRange(), 1.0)
    indices = DiskIndex(a, i, strategy)
    #@debug output_size, temparray_size, output_indices, temparray_indices, data_indices
    outputarray = create_outputarray(out, a, indices.output_size)
    # Read values from disk, depending on aliasing pattern
    outalias = output_aliasing(indices, ndims(outputarray), ndims(a))
    if outalias === :identical
        # We can read directly to the output array
        readblock_checked!(a, outputarray, indices.data_indices...)
    elseif outalias === :reshapeoutput
        # We need to reshape the output first, then read
        reshaped_output = reshape(outputarray, indices.temparray_size)
        readblock_checked!(a, reshaped_output, indices.data_indices...)
    else # :noalign
        # outputarray is only a subset of the chunk, so copy to temparray first
        temparray = Array{eltype(a)}(undef, indices.temparray_size...)
        readblock_checked!(a, temparray, indices.data_indices...)
        transfer_results_read!(outputarray, temparray, indices.output_indices, indices.temparray_indices)
    end
    return outputarray
end

"""
    setindex_disk!(A::AbstractArray, v, i...)

Internal `setindex!` for disk arrays.

Converts indices to ranges and calls `DiskArrays.writeblock!`
"""
function setindex_disk!(a::AbstractArray{T}, values::T, i...) where {T<:AbstractArray}
    checkscalar(a, i)
    # If values are not an array, wrap them in a vector
    return setindex_disk!(a, [values], i...)
end
function setindex_disk!(a::AbstractArray, values::AbstractArray, i...)
    # Check if we can write once or need to use multiple batches
    if need_batch(a, i)
        setindex_disk_batch!(a, values, i)
    else
        setindex_disk_nobatch!(a, values, i)
    end
    return values
end

function setindex_disk_batch!(a::AbstractArray, values::AbstractArray, i)
    # Convert indices to DiskIndex 
    indices = DiskIndex(a, i, batchstrategy(a))
    # Allocate a temporary array for writeblock
    temparray = Array{eltype(a)}(undef, indices.temparray_size...)
    # Get multi read indices for batches
    # TODO: this could be done more cleanlyu 
    # with a single object that accepts DiskIndex
    moutput_indices = MultiReadArray(indices.output_indices)
    mtemparray_indices = MultiReadArray(indices.temparray_indices)
    mdata_indicess = MultiReadArray(indices.data_indices)
    # Iterate over batches
    for ii in eachindex(moutput_indices)
        data_indices = mdata_indicess[ii]
        output_indices = moutput_indices[ii]
        temparray_indices = mtemparray_indices[ii]
        # Copy results to temparray
        transfer_results_write!(values, temparray, output_indices, temparray_indices)
        # Maybe view the array if we only need a subset
        vtemparray = maybeshrink(temparray, data_indices)
        if any(ind -> is_sparse_index(ind; density_threshold=1.0), temparray_indices)
            readblock!(a, vtemparray, data_indices...)
            transfer_results_write!(values, temparray, output_indices, temparray_indices)
        end
        # Write the results to a
        writeblock_checked!(a, vtemparray, data_indices...)
    end
    return nothing
end
function setindex_disk_nobatch!(a::AbstractArray, values::AbstractArray, i)
    # Get read and write indices
    indices = DiskIndex(a, i, NoBatch(batchstrategy(a)))
    # Write values to disk, depending on aliasing pattern
    outalias = output_aliasing(indices, ndims(a), ndims(values))
    if outalias === :identical
        # Nothing to do, write `values` directly to `a`
        writeblock_checked!(a, values, indices.data_indices...)
    elseif outalias === :reshapeoutput
        # First reshape, then write `reshaped_values` to `a`
        reshaped_values = reshape(values, indices.temparray_size)
        writeblock_checked!(a, reshaped_values, indices.data_indices...)
    else # :noalign
        # Blocks don't match, may need to read from `a` to `temparray` to fill gaps in the values
        temparray = Array{eltype(a)}(undef, indices.temparray_size...)
        if any(ind -> is_sparse_index(ind; density_threshold=1.0), indices.temparray_indices)
            # We have some sparse indexing pattern and are not in a batch situation, so
            # we need to read before writing.
            # This check could be optimized away in some cases, when writing unit ranges etc, 
            # but is probably not too expensive
            readblock!(a, temparray, indices.data_indices...)
        end
        # Copy `values` to `temparray`
        transfer_results_write!(values, temparray, indices.output_indices, indices.temparray_indices)
        # Write from `temparray` to `a`
        writeblock_checked!(a, temparray, indices.data_indices...)
    end
    return nothing
end

# Utils

"""
    create_outputarray(out, a, output_size)

Generate an `Array` to pass to `readblock!`
"""
function create_outputarray(out::AbstractArray, a::AbstractArray, output_size::Tuple)
    size(out) == output_size || throw(ArgumentError("Expected output array size of $output_size, got $(size(out))"))
    return out
end
create_outputarray(::Nothing, a::AbstractArray, output_size::Tuple) =
    Array{eltype(a)}(undef, output_size...)

"""
    transfer_results_read!(outputarray, temparray, outputindices, temparrayindices)

Copy results from `temparray` to `outputarray` for respective indices
"""
function transfer_results_read!(outputarray, temparray, outputindices, temparrayindices)
    outputarray[outputindices...] = view(temparray, temparrayindices...)
    outputarray
end
function transfer_results_read!(outputarray, temparray, oi::Tuple{Vararg{Int}}, ti::Tuple{Vararg{Int}})
    outputarray[oi...] = temparray[ti...]
    return outputarray
end

"""
    transfer_results_write!(values, temparray, valuesindices, temparrayindices)

Copy results from `values` to `temparry` for respective indices.
"""
function transfer_results_write!(values, temparray, valuesindices, temparrayindices)
    temparray[temparrayindices...] = view(values, valuesindices...)
    return temparray
end
function transfer_results_write!(values, temparray, vi::Tuple{Vararg{Int}}, ti::Tuple{Vararg{Int}})
    temparray[ti...] = values[oi...]
    return temparray
end

"""
    need_batch(a::AbstractArray, i) => Bool

Check if disk array `a` needs batch indexing for indices `i`, returning a `Bool`.
"""
Base.@assume_effects :foldable need_batch(a::AbstractArray, i) =
    _need_batch(eachchunk(a).chunks, i, batchstrategy(a))

function _need_batch(chunks, i, batch_strategy)
    needsbatch, chunksrem = _need_batch_index(first(i), chunks, batch_strategy)
    needsbatch ? true : _need_batch(chunksrem, Base.tail(i), batch_strategy)
end
_need_batch(::Tuple{}, ::Tuple{}, _) = false
_need_batch(::Tuple{}, _, _) = false
_need_batch(_, ::Tuple{}, _) = false

# Integer,UnitRange and Colon are contiguous and dont need batching
_need_batch_index(::Union{Integer,UnitRange,Colon}, chunks, _) = false, Base.tail(chunks)
# CartesianIndices are contiguous, also dont need batching, but chunks need splitting
_need_batch_index(i::CartesianIndices{N}, chunks, _) where {N} = false, last(splitchunks(i, chunks))
# CartesianIndex doesn't need batching, but chunks need splitting
_need_batch_index(i::CartesianIndex{N}, chunks, _) where {N} = false, last(splitchunks(i, chunks))
# StepRange doesn't need batching for CanStepRange strategies
_need_batch_index(::StepRange, chunks, ::BatchStrategy{CanStepRange}) = false, Base.tail(chunks)
# Everything else may need batching
function _need_batch_index(i, chunks, batchstrategy)
    chunksnow, chunksrem = splitchunks(i, chunks)
    allow_multi = allow_multi_chunk_access(batchstrategy)
    needsbatch = (allow_multi || has_chunk_gap(approx_chunksize.(chunksnow), i)) &&
                 is_sparse_index(i; density_threshold=density_threshold(batchstrategy))
    return needsbatch, chunksrem
end

"""
    maybeshrink(temparray::AbstractArray, indices::Tuple)

Shrink an array with a view, if needed.

TODO: this could be type stable if we reshaped the array instead.
"""
function maybeshrink(temparray::AbstractArray, indices::Tuple)
    if all(size(temparray) .== length.(indices))
        temparray
    else
        view(temparray, map(i -> 1:length(i), indices)...)
    end
end

"Like `readblock!`, but only exectued when data size to read is not empty"
function readblock_checked!(a::AbstractArray, out::AbstractArray, i...)
    if length(out) > 0
        readblock!(a, out, i...)
    end
end

"Like `writeblock!`, but only exectued when data size to read is not empty"
function writeblock_checked!(a::AbstractArray, values::AbstractArray, i...)
    if length(values) > 0
        writeblock!(a, values, i...)
    end
end


# Implementation macros

macro implement_getindex(t)
    t = esc(t)
    quote
        DiskArrays.isdisk(::Type{<:$t}) = true
        Base.getindex(a::$t, i...) = getindex_disk(a, i...)
        function DiskArrays.ChunkIndices(a::$t; offset=false)
            return ChunkIndices(
                map(s -> 1:s, size(eachchunk(a))), offset ? OffsetChunks() : OneBasedChunks()
            )
        end
    end
end

macro implement_setindex(t)
    t = esc(t)
    quote
        Base.setindex!(a::$t, v::AbstractArray, i...) = setindex_disk!(a, v, i...)

        # Add an extra method if a single number is given
        function Base.setindex!(a::$t{<:Any,N}, v, i...) where {N}
            return Base.setindex!(a, fill(v, ntuple(i -> 1, N)...), i...)
        end

        function Base.setindex!(a::$t, v::AbstractArray, i::ChunkIndex)
            cs = eachchunk(a)
            inds = cs[i.I]
            return setindex_disk!(a, v, inds...)
        end
    end
end
