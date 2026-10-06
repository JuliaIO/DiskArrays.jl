"""
    IterBlocks{N}

The blocks that iteration reads a disk array in: whole dimensions before `k`, chunk ranges
along `k`, single indices after it. Each block is contiguous in column-major order, so
iterating its values in turn yields the array in Base's iteration order, which consumers
like `zip`, generators and `enumerate` rely on.

`k` is the last dimension whose blocks fit in `default_chunk_size` megabytes. When that is
`N`, every chunk is read once; a smaller `k` bounds memory by re-reading the chunks that span
several indices after `k`.
"""
struct IterBlocks{N}
    size::NTuple{N,Int}
    k::Int
    kranges::Vector{UnitRange{Int}}
    grid::CartesianIndices{N,NTuple{N,Base.OneTo{Int}}}
end

function IterBlocks(a::AbstractArray{<:Any,N}) where {N}
    chunks = eachchunk(a).chunks
    sz = size(a)
    limit = default_chunk_size[] * 1_000_000 / element_size(a)
    k, before = 1, 1
    for d in 1:N
        before * max_chunksize(chunks[d]) <= limit || break
        k = d
        before *= sz[d]
    end
    kranges = collect(UnitRange{Int}, chunks[k])
    grid = CartesianIndices(ntuple(d -> d < k ? 1 : d == k ? length(kranges) : sz[d], Val(N)))
    return IterBlocks{N}(sz, k, kranges, grid)
end

function _block(b::IterBlocks{N}, I::CartesianIndex{N}) where {N}
    return ntuple(Val(N)) do d
        d < b.k ? (1:b.size[d]) : d == b.k ? b.kranges[I[d]] : (I[d]:I[d])
    end
end

_readblock(a::AbstractArray{T,N}, inds) where {T,N} = vec(convert(Array{T,N}, a[inds...]))

# State: the blocks, the values of the current block, the position in it and the grid state.
function _iterate_disk(a::AbstractArray)
    # A zero-length dimension still has one, empty, chunk
    isempty(a) && return nothing
    blocks = IterBlocks(a)
    return _iterate_block(a, blocks, iterate(blocks.grid))
end
function _iterate_disk(a::AbstractArray, (blocks, values, i, gridstate))
    i < length(values) && return values[i+1], (blocks, values, i + 1, gridstate)
    return _iterate_block(a, blocks, iterate(blocks.grid, gridstate))
end
_iterate_disk(a::AbstractArray{<:Any,0}, i=1) = i == 1 ? (@inbounds a[i], 2) : nothing

function _iterate_block(a::AbstractArray, blocks::IterBlocks, g)
    isnothing(g) && return nothing
    I, gridstate = g
    values = _readblock(a, _block(blocks, I))
    return first(values), (blocks, values, 1, gridstate)
end

macro implement_iteration(t)
    t = esc(t)
    quote
        Base.iterate(a::$t) = _iterate_disk(a)
        Base.iterate(a::$t, i) = _iterate_disk(a, i)
    end
end
