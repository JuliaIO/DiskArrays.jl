"""
    AllowStepRange

Traits to specify if an array axis can utilise step ranges,
as an argument to `BatchStrategy` types `NoBatch`, `SubRanges`
and `ChunkRead`.

`CanStepRange()` and `NoStepRange()` are the two options.
"""
abstract type AllowStepRange end

struct CanStepRange <: AllowStepRange end
struct NoStepRange <: AllowStepRange end

"""
    BatchStrategy{S<:AllowStepRange}
    
Traits for array chunking strategy.

`NoBatch`, `SubRanges` and `ChunkRead` are the options.

All have keywords:

- `alow_steprange`: an [`AllowStepRange`](@ref) trait, NoStepRange() by default.
    this controls if step range are passed to the parent object.
- `density_threshold`: determines the density where step ranges are not read as whole chunks.
"""
abstract type BatchStrategy{S<:AllowStepRange} end

"""
    NoBatch <: BatchStrategy

A chunking strategy that avoids batching into multiple reads.
"""
@kwdef struct NoBatch{S} <: BatchStrategy{S}
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
NoBatch(from::BatchStrategy) =
    NoBatch(from.allow_steprange, from.density_threshold)

"""
    SubRanges <: BatchStrategy

A chunking strategy that splits contiguous streaks 
into ranges to be read separately.
"""
@kwdef struct SubRanges{S} <: BatchStrategy{S}
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
SubRanges(from::BatchStrategy) =
    SubRanges(from.allow_steprange, from.density_threshold)
"""
    ChunkRead <: BatchStrategy

A chunking strategy splits a dataset according to chunk,
and reads chunk by chunk.
"""
@kwdef struct ChunkRead{S} <: BatchStrategy{S}
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
ChunkRead(from::BatchStrategy) =
    ChunkRead(from.allow_steprange, from.density_threshold)

batchstrategy(x) = batchstrategy(haschunks(x))

allow_steprange(a) = allow_steprange(batchstrategy(a))
allow_steprange(::BatchStrategy{S}) where {S} = allow_steprange(S)
allow_steprange(::Type{CanStepRange}) = true
allow_steprange(::Type{NoStepRange}) = false
allow_steprange(::CanStepRange) = true
allow_steprange(::NoStepRange) = false

allow_multi_chunk_access(a) = allow_multi_chunk_access(batchstrategy(a))
allow_multi_chunk_access(::ChunkRead) = false
allow_multi_chunk_access(::SubRanges) = true

density_threshold(a) = density_threshold(batchstrategy(a))
density_threshold(a::BatchStrategy) = a.density_threshold


# Utils

function has_chunk_gap(chunksize, ids::AbstractVector{<:Integer})
    # Find largest jump in indices
    isempty(ids) && return false
    minind, maxind = extrema(ids)
    maxind - minind > first(chunksize)
end
# Return true for all multidimensional indices for now, could be optimised in the future
has_chunk_gap(chunksize, ids) = true

# Compute the number of possible indices in the hyperrectangle
function span(a::AbstractArray{<:Integer})
    iszero(length(a)) ? 0 : 1 - (-(extrema(a)...))
end
function span(a::AbstractArray{CartesianIndex{N}}) where {N}
    minind, maxind = extrema(a)
    prod((maxind - minind + oneunit(minind)).I)
end
function span(a::AbstractArray{Bool})
    minind, maxind = extrema(view(CartesianIndices(size(a)), a))
    prod((maxind - minind + oneunit(minind)).I)
end
# The number of indices to actually be read
numind(a::AbstractArray{Bool}) = sum(a)
numind(a::Union{AbstractArray{<:Integer},AbstractArray{<:CartesianIndex}}) = length(a)

function is_sparse_index(ids; density_threshold=0.5)
    indexdensity = numind(ids) / span(ids)
    return indexdensity < density_threshold
end

function find_subranges_sorted(inds, allow_steprange=false)
    T = allow_steprange ? Union{UnitRange{Int},StepRange{}} : UnitRange{Int}
    rangelist = T[]
    outputinds = UnitRange{Int}[]
    current_step = 0
    current_base = 1
    for iind in 1:length(inds)-1
        next_step = inds[iind+1] - inds[iind]
        if (next_step == current_step) || (next_step == 0)
            nothing
        else
            if !allow_steprange && next_step != 1
                #Need to close the range
                push!(rangelist, inds[current_base]:inds[iind])
                push!(outputinds, current_base:iind)
                current_base = iind + 1
                current_step = 0
                continue
            end
            if current_step === 0
                # Just set the step (hasnt been set before)
                current_step = inds[iind+1] - inds[iind]
            else
                #Need to close the range
                if current_step == 1
                    push!(rangelist, inds[current_base]:inds[iind])
                else
                    push!(rangelist, inds[current_base]:current_step:inds[iind])
                end
                push!(outputinds, current_base:iind)
                current_base = iind + 1
                current_step = 0
                continue
            end
        end
    end
    if current_step == 1 || current_step == 0
        push!(rangelist, inds[current_base]:last(inds))
    else
        push!(rangelist, inds[current_base]:current_step:last(inds))
    end
    push!(outputinds, current_base:length(inds))

    return rangelist, outputinds
end

# For index arrays >1D we need to store the cartesian indices in the sort
# perm result
function mysortperm(i)
    p = collect(vec(CartesianIndices(i)))
    sort!(p; by=Base.Fix1(getindex, i))
    p
end
mysortperm(i::AbstractVector) = sortperm(i)


function process_index(i, chunks::Tuple{Vararg{ChunkVector}}, strategy::Union{ChunkRead,SubRanges})
    ii, chunksrem = process_index(i, chunks, NoBatch(strategy))
    di = DiskIndex(
        ii.output_size,
        ii.temparray_size,
        ([ii.output_indices],),
        ([ii.temparray_indices],),
        ([ii.data_indices],)
    )
    return di, chunksrem
end
function process_index(i::AbstractArray{<:Integer,N}, chunks::Tuple{Vararg{ChunkVector}}, ::ChunkRead) where {N}
    chunksdict = Dict{Int,Vector{Pair{Int,CartesianIndex{N}}}}()
    # Look for affected chunks
    for outindex in CartesianIndices(i)
        dataindex = i[outindex]
        chunkindex = findchunk(first(chunks), dataindex)
        indexpairs = get!(() -> Pair{Int,Int}[], chunksdict, chunkindex)
        push!(indexpairs, dataindex => outindex)
    end
    # Allocate empty index vectors
    tempinds = Tuple{Vector{Int}}[]
    datainds = Tuple{UnitRange{Int}}[]
    outinds = Tuple{Vector{CartesianIndex{N}}}[]
    maxtempindex = -1
    # Calculate data, temp and output indices
    for (chunkindex, indexpairs) in chunksdict
        dataindex = extrema(first, indexpairs)
        tempindex = first.(indexpairs) .- first(dataindex) .+ 1
        # Push indices 
        push!(outinds, (map(last, indexpairs),))
        push!(datainds, (first(dataindex):last(dataindex),))
        push!(tempinds, (tempindex,))
        # Track the maximum tempindex
        maxtempindex = max(maxtempindex, maximum(tempindex))
    end
    di = DiskIndex(size(i), ((maxtempindex),), (outinds,), (tempinds,), (datainds,))
    return di, Base.tail(chunks)
end
# Implement NCDatasets behavior of splitting list of indices into ranges
function process_index(i::AbstractArray{<:Integer,N}, chunks::Tuple{Vararg{ChunkVector}}, s::SubRanges) where {N}
    di = if i isa AbstractVector && issorted(i)
        rangelist, outputinds = find_subranges_sorted(i, allow_steprange(s))
        datainds = tuple.(rangelist)
        tempinds = map(rangelist, outputinds) do rl, oi
            v = view(i, oi)
            r = map(x -> (x - first(v)) ÷ step(rl) + 1, v)
            (r,)
        end
        outinds = tuple.(outputinds)
        tempsize = maximum(length, rangelist)
        DiskIndex((length(i),), (tempsize,), (outinds,), (tempinds,), (datainds,))
    else
        p = mysortperm(i)
        i_sorted = view(i, p)
        rangelist, outputinds = find_subranges_sorted(i_sorted, allow_steprange(s))
        datainds = tuple.(rangelist)
        tempinds = map(rangelist, outputinds) do rl, oi
            v = view(i_sorted, oi)
            r = map(x -> (x - first(v)) ÷ step(rl) + 1, v)
            (r,)
        end
        outinds = map(outputinds) do oi
            (view(p, oi),)
        end
        tempsize = maximum(length(rangelist))
        DiskIndex(size(i), (tempsize,), (outinds,), (tempinds,), (datainds,))
    end
    return di, Base.tail(chunks)
end
function process_index(i::AbstractArray{Bool,N}, chunks::Tuple{Vararg{ChunkVector}}, cr::ChunkRead) where {N}
    process_index(findall(i), chunks, cr)
end
function process_index(i::AbstractArray{Bool,N}, chunks::Tuple{Vararg{ChunkVector}}, cr::SubRanges) where {N}
    process_index(findall(i), chunks, cr)
end
function process_index(i::StepRange{<:Integer}, chunks::Tuple{Vararg{ChunkVector}}, ::ChunkRead{CanStepRange})
    di = DiskIndex((length(i),), (length(i),), ([(Colon(),)],), ([(Colon(),)],), ([(i,)],))
    return di, Base.tail(chunks)
end
function process_index(i::StepRange{<:Integer}, chunks::Tuple{Vararg{ChunkVector}}, ::SubRanges{CanStepRange})
    di = DiskIndex((length(i),), (length(i),), ([(Colon(),)],), ([(Colon(),)],), ([(i,)],))
    return di, Base.tail(chunks)
end
function process_index(i::StepRange{<:Integer}, chunks::Tuple{Vararg{ChunkVector}}, ::NoBatch{CanStepRange})
    di = DiskIndex((length(i),), (length(i),), (Colon(),), (Colon(),), (i,))
    return di, Base.tail(chunks)
end
function process_index(
    i::AbstractArray{<:CartesianIndex{N},M}, chunks::Tuple{Vararg{ChunkVector}}, ::Union{ChunkRead,SubRanges}
) where {N,M}
    chunksnow, chunksrem = splitchunks(i, chunks)
    chunksdict = Dict{CartesianIndex{N},Vector{Pair{CartesianIndex{N},CartesianIndex{M}}}}()
    # Look for affected chunks
    for outindex in CartesianIndices(i)
        dataindex = i[outindex]
        cI = CartesianIndex(findchunk.(chunksnow, dataindex.I))
        a = get!(() -> Pair{CartesianIndex{N},CartesianIndex{M}}[], chunksdict, cI)
        push!(a, dataindex => outindex)
    end

    tempinds = Tuple{Vector{CartesianIndex{N}}}[]
    datainds = NTuple{N,UnitRange{Int}}[]
    outinds = Tuple{Vector{CartesianIndex{M}}}[]
    tempsize = map(_ -> 0, chunksnow)
    for (cI, a) in chunksdict
        datamin, datamax = extrema(first, a)
        aa = first.(a)
        tempind = map(aa) do ind
            ind - datamin + oneunit(CartesianIndex{N})
        end
        push!(outinds, tuple(map(last, a)))
        push!(datainds, range.(datamin.I, datamax.I))
        push!(tempinds, tuple(tempind))
        s = datamax.I .- datamin.I .+ 1
        tempsize = max.(s, tempsize)
    end
    di = DiskIndex(size(i), tempsize, (outinds,), (tempinds,), (datainds,))

    return di, chunksrem
end
