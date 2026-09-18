"""
    RangeIndex(ranges...)

Index a disk array with a list of ranges. Each range is read (or written) as one block,
instead of the hull `minimum:maximum` or a chunk-by-chunk scan.

It is the `AbstractVector{Int}` of the concatenated ranges, so wrappers such as
DimensionalData.jl pass it through and can slice their lookups with it.
"""
struct RangeIndex <: AbstractVector{Int}
    ranges::Vector{UnitRange{Int}}
end
RangeIndex(ranges::AbstractUnitRange{<:Integer}...) = RangeIndex(UnitRange{Int}[ranges...])

Base.size(i::RangeIndex) = (sum(length, i.ranges; init=0),)
function Base.getindex(i::RangeIndex, k::Int)
    @boundscheck checkbounds(i, k)
    for r in i.ranges
        k <= length(r) && return r[k]
        k -= length(r)
    end
    throw(BoundsError(i, k))
end
# Slicing keeps it a list of ranges, so reads through a `view` still go block by block
function Base.getindex(i::RangeIndex, k::AbstractUnitRange{<:Integer})
    @boundscheck checkbounds(i, k)
    ranges = UnitRange{Int}[]
    offset = 0
    for r in i.ranges
        lo, hi = max(first(k), offset + 1), min(last(k), offset + length(r))
        lo <= hi && push!(ranges, r[lo-offset]:r[hi-offset])
        offset += length(r)
    end
    return RangeIndex(ranges)
end
Base.show(io::IO, i::RangeIndex) = print(io, "RangeIndex(", join(i.ranges, ", "), ")")
Base.show(io::IO, ::MIME"text/plain", i::RangeIndex) = show(io, i)

# Always batch, grouping the ranges into blocks. `NoBatch` keeps the
# `AbstractArray{<:Integer}` behaviour, a single read of the hull.
_need_batch_index(::RangeIndex, chunks, ::Union{ChunkRead,SubRanges}) = true, tail(chunks)
for S in (:ChunkRead, :SubRanges) # one method per strategy, to beat the `AbstractArray{<:Integer}` ones
    @eval function process_index(i::RangeIndex, chunks::Tuple{Vararg{ChunkVector}}, ::$S)
        stops = cumsum(map(length, i.ranges))
        outputs = map((r, stop) -> stop-length(r)+1:stop, i.ranges, stops)
        groups = group_ranges(i.ranges, first(chunks))
        datainds = map(g -> (minimum(first, i.ranges[g]):maximum(last, i.ranges[g]),), groups)
        outinds = map(g -> (RangeIndex(outputs[g]),), groups)
        tempinds = map((g, d) -> (RangeIndex(map(r -> r .- (first(d[1]) - 1), i.ranges[g])),), groups, datainds)
        tempsize = maximum(d -> length(d[1]), datainds; init=0)
        di = DiskIndex((length(i),), (tempsize,), (outinds,), (tempinds,), (datainds,))
        return di, tail(chunks)
    end
end

"""
    group_ranges(ranges, chunks::ChunkVector) => Vector{Vector{Int}}

Group `ranges` (by position) so that each group is read as one block, with as few
blocks as possible at no extra chunk I/O: consecutive ranges, sorted by start, are
split only when a whole chunk lies in the gap between them.
"""
function group_ranges(ranges, chunks::ChunkVector)
    groups = Vector{Int}[]
    stop = 0 # last data index of the current group
    for k in sortperm(ranges; by=first)
        r = ranges[k]
        isempty(r) && continue
        if !isempty(groups) && findchunk(chunks, first(r)) - findchunk(chunks, stop) <= 1
            push!(groups[end], k)
        else
            push!(groups, [k])
        end
        stop = max(stop, last(r))
    end
    return groups
end
