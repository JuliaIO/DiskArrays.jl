"""
    IterBlocks{N}

The plan that iteration reads a disk array by. Base iterates arrays in column-major order, so
iteration reads blocks that are contiguous in that order: whole dimensions before `k`, a range
along `k`, single indices after it. Iterating the values of each block in turn then yields the
array in Base's order, which consumers like `zip`, generators and `enumerate` rely on.

`k` is the first dimension that does not fit whole, with the dimensions before it, in the
memory budget: `default_chunk_size` megabytes, or one chunk if that is larger. The ranges along
`k` are its chunks, each split into equal pieces when it does not fit in the budget beside the
leading dimensions. Each chunk is then read once per piece and once per index after `k` that it
spans; when `k` is the last dimension and the chunks fit, every chunk is read once.

A block is filled lazily, one chunk at a time, when iteration reaches the first value that
chunk holds, so stopping early reads only the chunks already reached.
"""
# Mutable only to be a reference: copying it into every iteration state is slow
mutable struct IterBlocks{N}
    size::NTuple{N,Int}
    k::Int
    # Ranges along `k`
    pieces::Vector{UnitRange{Int}}
    # The chunks of a block, as their ranges in the dimensions before `k`, in column-major order
    leading::Vector{NTuple{N,UnitRange{Int}}}
    # The linear position in a block of the first value of each of `leading`, then `typemax(Int)`
    loadpos::Vector{Int}
    grid::CartesianIndices{N,NTuple{N,Base.OneTo{Int}}}
end

function IterBlocks(a::AbstractArray{<:Any,N}) where {N}
    chunks = map(c -> collect(UnitRange{Int}, c), eachchunk(a).chunks)
    sz = size(a)
    # Reading a chunk takes that much memory anyway
    budget = max(
        floor(Int, default_chunk_size[] * 1_000_000 / element_size(a)),
        prod(c -> maximum(length, c), chunks),
    )
    k, before = _iterdim(sz, budget)
    fits = budget ÷ before
    pieces = UnitRange{Int}[]
    for r in chunks[k]
        n = cld(length(r), fits)
        for i in 1:n
            push!(pieces, (first(r) + (i - 1) * length(r) ÷ n):(first(r) + i * length(r) ÷ n - 1))
        end
    end
    strides = (1, Base.front(cumprod(sz))...)
    nleading = ntuple(d -> d < k ? length(chunks[d]) : 1, Val(N))
    leading = vec(map(CartesianIndices(nleading)) do J
        ntuple(d -> d < k ? chunks[d][J[d]] : (1:1), Val(N))
    end)
    loadpos = map(leading) do rs
        1 + sum(d -> d < k ? (first(rs[d]) - 1) * strides[d] : 0, 1:N)
    end
    push!(loadpos, typemax(Int))
    grid = CartesianIndices(ntuple(d -> d < k ? 1 : d == k ? length(pieces) : sz[d], Val(N)))
    return IterBlocks{N}(sz, k, pieces, leading, loadpos, grid)
end

# The first dimension that does not fit whole after the ones before it, or the last, and the
# length of those before it
function _iterdim(sz::NTuple{N,Int}, budget::Int) where {N}
    before = 1
    for d in 1:N-1
        before * sz[d] > budget && return d, before
        before *= sz[d]
    end
    return N, before
end

# The ranges of the `n`th block in the array, and of its `s`th chunk in the array and in the block
function _blockranges(b::IterBlocks{N}, n::Int) where {N}
    I = b.grid[n]
    return ntuple(Val(N)) do d
        d < b.k ? (1:b.size[d]) : d == b.k ? b.pieces[I[d]] : (I[d]:I[d])
    end
end
function _chunkranges(b::IterBlocks{N}, n::Int, s::Int) where {N}
    I = b.grid[n]
    rs = b.leading[s]
    src = ntuple(d -> d < b.k ? rs[d] : d == b.k ? b.pieces[I[d]] : (I[d]:I[d]), Val(N))
    dst = ntuple(d -> d < b.k ? rs[d] : d == b.k ? (1:length(b.pieces[I[d]])) : (1:1), Val(N))
    return src, dst
end

function _newblock(a::AbstractArray{T,N}, b::IterBlocks{N}, n::Int) where {T,N}
    src, _ = _chunkranges(b, n, 1)
    # A block of one chunk is the values read
    length(b.leading) == 1 && return convert(Array{T,N}, a[src...])
    buf = Array{T,N}(undef, map(length, _blockranges(b, n)))
    _loadchunk!(buf, a, b, n, 1)
    return buf
end
function _loadchunk!(buf::Array, a::AbstractArray, b::IterBlocks, n::Int, s::Int)
    src, dst = _chunkranges(b, n, s)
    copyto!(view(buf, dst...), a[src...])
    return buf
end

# State: the plan, the values of the current block, the position in them, the last position
# before the next chunk to read, that chunk, and the linear index of the block. Each block has
# its own `values` and reading a chunk only writes its own values, so an earlier state stays
# valid. The state holds only references and integers, which keeps the loop fast.
function _iterate_disk(a::AbstractArray)
    # A zero-length dimension still has one, empty, chunk
    isempty(a) && return nothing
    blocks = IterBlocks(a)
    values = _newblock(a, blocks, 1)
    return first(values), (blocks, values, 1, _stop(blocks, values, 1), 2, 1)
end
@inline function _iterate_disk(a::AbstractArray, state)
    blocks, values, i, stop, s, n = state
    i < stop && return (@inbounds values[i+1]), (blocks, values, i + 1, stop, s, n)
    # A concrete return type keeps the state unboxed in the caller's loop
    ok, state = _iterate_advance(a, blocks, values, i, s, n)
    ok || return nothing
    return (@inbounds state[2][state[3]]), state
end
_iterate_disk(a::AbstractArray{<:Any,0}, i=1) = i == 1 ? (@inbounds a[i], 2) : nothing

# Read the next chunk of the block, or the next block, and point the state at its first value
@noinline function _iterate_advance(a::AbstractArray, blocks, values, i, s, n)
    if i < length(values)
        _loadchunk!(values, a, blocks, n, s)
        return true, (blocks, values, i + 1, _stop(blocks, values, s), s + 1, n)
    end
    n < length(blocks.grid) || return false, (blocks, values, i, i, s, n)
    values = _newblock(a, blocks, n + 1)
    return true, (blocks, values, 1, _stop(blocks, values, 1), 2, n + 1)
end

# The last position before the chunk after `s`; `loadpos` ends with `typemax(Int)`
_stop(blocks::IterBlocks, values, s) = min(blocks.loadpos[s+1] - 1, length(values))

macro implement_iteration(t)
    t = esc(t)
    quote
        Base.iterate(a::$t) = _iterate_disk(a)
        Base.iterate(a::$t, i) = _iterate_disk(a, i)
    end
end
