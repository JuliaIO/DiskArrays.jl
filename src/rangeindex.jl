"""
    RangeIndex(ranges...)

Index a disk array with a sorted list of non-empty, non-overlapping ranges. Sparse enough
to be batched, they are read (or written) in blocks shaped by the array's batch strategy:
one block per chunk touched for `ChunkRead`, one block per range for `SubRanges`; adjacent
ranges always join.

It is the `AbstractVector{Int}` of the concatenated ranges, so wrappers such as
DimensionalData.jl pass it through and can slice their lookups with it.
"""
struct RangeIndex <: AbstractVector{Int}
    ranges::Vector{UnitRange{Int}}
    stops::Vector{Int} # cumulative lengths, for `O(log n)` `getindex`
    function RangeIndex(ranges::Vector{UnitRange{Int}})
        all(!isempty, ranges) && all(k -> last(ranges[k]) < first(ranges[k+1]), 1:length(ranges)-1) ||
            throw(ArgumentError("ranges must be non-empty, sorted and non-overlapping, got $ranges"))
        return new(ranges, cumsum(map(length, ranges)))
    end
end
RangeIndex(ranges::AbstractUnitRange{<:Integer}...) = RangeIndex(UnitRange{Int}[ranges...])

Base.size(i::RangeIndex) = (isempty(i.stops) ? 0 : last(i.stops),)
function Base.getindex(i::RangeIndex, k::Int)
    @boundscheck checkbounds(i, k)
    j = searchsortedfirst(i.stops, k)
    r = i.ranges[j]
    return r[k - i.stops[j] + length(r)]
end
# Slicing keeps it a list of ranges, so reads through a `view` still go block by block
function Base.getindex(i::RangeIndex, k::AbstractUnitRange{<:Integer})
    @boundscheck checkbounds(i, k)
    isempty(k) && return RangeIndex()
    j1, j2 = searchsortedfirst(i.stops, first(k)), searchsortedfirst(i.stops, last(k))
    ranges = i.ranges[j1:j2]
    ranges[1] = i[first(k)]:last(ranges[1])
    ranges[end] = first(ranges[end]):i[last(k)]
    return RangeIndex(ranges)
end
# Sorted, so the batching decision (`has_chunk_gap`, `is_sparse_index`) is `O(1)`
Base.minimum(i::RangeIndex) = first(first(i.ranges))
Base.maximum(i::RangeIndex) = last(last(i.ranges))
Base.extrema(i::RangeIndex) = (minimum(i), maximum(i))
Base.show(io::IO, i::RangeIndex) = print(io, "RangeIndex(", join(i.ranges, ", "), ")")
Base.show(io::IO, ::MIME"text/plain", i::RangeIndex) = show(io, i)
Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::RangeIndex) =
    isempty(i.ranges) || (first(i.ranges[1]) >= first(inds) && last(i.ranges[end]) <= last(inds))
# In-memory arrays and ranges (temp buffers, lookups) index fastest with a plain `Vector{Int}`,
# so resolve to one there. Disk arrays and wrappers never see this and keep the `RangeIndex`.
Base.to_indices(A::Union{DenseArray,AbstractRange}, inds, I::Tuple{RangeIndex,Vararg}) =
    (collect(I[1]), to_indices(A, Base.safe_tail(inds), tail(I))...)

# Batched exactly when the same indices as a vector would be (`_need_batch_index`); dense
# enough for the strategy's `density_threshold`, the hull is read in one block. Batched, the
# blocks are ranges rather than chunk hulls or contiguous runs.
for S in (:ChunkRead, :SubRanges) # one method per strategy, to beat the `AbstractArray{<:Integer}` ones
    @eval function process_index(i::RangeIndex, chunks::Tuple{Vararg{ChunkVector}}, s::$S)
        groups = group_ranges(i.ranges, first(chunks), s)
        datainds = map(g -> (first(i.ranges[first(g)]):last(i.ranges[last(g)]),), groups)
        outinds = map(g -> (i.stops[first(g)]-length(i.ranges[first(g)])+1:i.stops[last(g)],), groups)
        # Within a block the temp array is read contiguously, or through the shifted ranges if there are gaps
        tempinds = map(groups, datainds, outinds) do g, d, o
            length(d[1]) == length(o[1]) ? (1:length(o[1]),) : (RangeIndex(map(r -> r .- (first(d[1]) - 1), i.ranges[g])),)
        end
        tempsize = maximum(d -> length(d[1]), datainds; init=0)
        di = DiskIndex((length(i),), (tempsize,), (outinds,), (tempinds,), (datainds,))
        return di, tail(chunks)
    end
end

"""
    group_ranges(ranges, chunks::ChunkVector, strategy::BatchStrategy) => Vector{UnitRange{Int}}

Group sorted `ranges` (by position) into blocks, each read with one `readblock!`.
Adjacent ranges always join. `ChunkRead` also joins ranges that touch the same chunk,
which is read anyway; `SubRanges` backends have cheap random access and their own cache,
so every other range is read on its own.
"""
function group_ranges(ranges, chunks::ChunkVector, strategy::BatchStrategy)
    groups = UnitRange{Int}[]
    for k in eachindex(ranges)
        if k > 1 && joins(ranges[k-1], ranges[k], chunks, strategy)
            groups[end] = first(groups[end]):k
        else
            push!(groups, k:k)
        end
    end
    return groups
end
joins(a, b, chunks, ::ChunkRead) = last(a) + 1 == first(b) || findchunk(chunks, last(a)) == findchunk(chunks, first(b))
joins(a, b, chunks, ::SubRanges) = last(a) + 1 == first(b)
