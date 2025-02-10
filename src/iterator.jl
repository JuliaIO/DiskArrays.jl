using OffsetArrays: OffsetArray

"""
    BlockedIndices{C<:GridChunks}

A lazy iterator over the indices of GridChunks.

Uses two `Iterators.Stateful` iterators, at chunk and indices levels.
"""
struct BlockedIndices{C<:GridChunks}
    gridchunks::C
end

# Base methods

Base.length(b::BlockedIndices) = prod(last.(last.(b.gridchunks.chunks)))
Base.IteratorEltype(::Type{<:BlockedIndices}) = Base.HasEltype()
Base.IteratorSize(::Type{<:BlockedIndices{<:GridChunks{N}}}) where {N} = Base.HasShape{N}()
Base.size(b::BlockedIndices)::NTuple{<:Any,Int} = map(last ∘ last, b.gridchunks.chunks)
Base.eltype(b::BlockedIndices) = CartesianIndex{ndims(b.gridchunks)}

function Base.iterate(a::BlockedIndices)
    # Define an outer iterator over chunks
    chunkstate = Iterators.Stateful(a.gridchunks)
    # And iterate it once
    ic = iterate(chunkstate)
    # Exit early if there are no chunks at all
    isnothing(ic) && return nothing
    # Define an inner iterator over chunk indices
    innerstate = Iterators.Stateful(CartesianIndices(first(ic)))
    # Iterate it once
    i = iterate(innerstate)
    # Again exit if its empty
    isnothing(i) && return nothing
    # Otherwise return the first index and both iterators
    state = (chunkstate, innerstate)
    return first(i), state
end
function Base.iterate(::BlockedIndices, state)
    chunkstate, innerstate = state
    i = iterate(innerstate)
    if isnothing(i)
        c = iterate(chunkstate)
        # There are no more chunks, exit
        isnothing(c) && return nothing
        # Set the inner iterator to the start of the next chunk
        Iterators.reset!(innerstate, CartesianIndices(first(c)))
        i = iterate(innerstate)
        # This chunk is empty, exit
        isnothing(i) && return nothing
        # Return the next index and iterator state
        state = (chunkstate, innerstate)
        return first(i), state
    else
        # Return the next index and iterator state
        state = (chunkstate, innerstate)
        return first(i), state
    end
end

# Implementaion macros

# Nested iteration over chunks
@noinline function _iterate_disk(
    a::AbstractArray{T}, i::I
) where {T,I<:Tuple{A,B,C}} where {A,B,C}
    # Split the data, block indices and state from the iterator
    currentdata::A, blockinds::B, state::C = i
    # And split the block stat into the chunk iterator and inner indices
    (chunkstate, innerstate) = state
    # Need to check now as state will be updated
    innerstate_was_empty = isempty(innerstate)
    # Iterate over the block indices
    blockiter = iterate(blockinds, state)
    # Check if we reached the end
    if isnothing(blockiter)
        return nothing
    else
        # Get the next index and updated iterator state
        i, newstate = blockiter
        if innerstate_was_empty
            # We need to move to a new chunk
            (newchunkstate, newinnerstate) = newstate
            newchunk = newinnerstate.itr.indices
            # Get a new chunk of data
            newdata = OffsetArray(a[newchunk...], newinnerstate.itr)
            return newdata[i]::T, (newdata, blockinds, newstate)::I
        else
            # Current chunk still has values left to iterate over
            return currentdata[i]::T, (currentdata, blockinds, newstate)::I
        end
    end
end
@noinline function _iterate_disk(a)
    # Get the indices for each chunk of data
    blockinds = BlockedIndices(eachchunk(a))
    iterator = iterate(blockinds)
    # No chunks at all, early exit
    isnothing(iterator) && return nothing
    i, state = iterator
    (chunkstate, innerstate) = state
    currentchunk = innerstate.itr.indices
    # Get the first chunk of data to iterate over
    currentdata = OffsetArray(a[currentchunk...], innerstate.itr)
    return currentdata[i], (currentdata, blockinds, state)
end

macro implement_iteration(t)
    t = esc(t)
    quote
        Base.eachindex(a::$t) = BlockedIndices(eachchunk(a))
        Base.iterate(a::$t) = _iterate_disk(a)
        Base.iterate(a::$t, i) = _iterate_disk(a, i)
    end
end
