import Base.Broadcast: Broadcasted, AbstractArrayStyle, DefaultArrayStyle, flatten

# DiskArrays broadcast style

struct ChunkStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

Base.BroadcastStyle(::ChunkStyle{N}, ::ChunkStyle{M}) where {N,M} = ChunkStyle{max(N, M)}()
function Base.BroadcastStyle(::ChunkStyle{N}, ::DefaultArrayStyle{M}) where {N,M}
    return ChunkStyle{max(N, M)}()
end
function Base.BroadcastStyle(::DefaultArrayStyle{M}, ::ChunkStyle{N}) where {N,M}
    return ChunkStyle{max(N, M)}()
end

struct BroadcastDiskArray{T,N,BC<:Broadcasted{<:ChunkStyle{N}}} <: AbstractDiskArray{T,N}
    broadcasted::BC
end
function BroadcastDiskArray(broadcasted::B) where {B<:Broadcasted{<:ChunkStyle{N}}} where {N}
    ElType = Base.Broadcast.combine_eltypes(broadcasted.f, broadcasted.args)
    return BroadcastDiskArray{ElType,N,B}(broadcasted)
end

# Base methods

Base.size(a::BroadcastDiskArray) = size(a.broadcasted)
Base.broadcastable(a::BroadcastDiskArray) = a.broadcasted
Base.copy(a::BroadcastDiskArray) = copyto!(zeros(eltype(a), size(a)), a.broadcasted)

Base.copy(broadcasted::Broadcasted{ChunkStyle{N}}) where {N} =
    BroadcastDiskArray(flatten(broadcasted))
function Base.copyto!(dest::AbstractArray, broadcasted::Broadcasted{ChunkStyle{N}}) where {N}
    bcf = flatten(broadcasted)
    # Get a list of chunks to apply
    gcd = common_chunks(size(bcf), dest, bcf.args...)
    # Apply the broadcast to dest chunk by chunk
    foreach(gcd) do chunk
        # Possible optimization would be to use a LRU cache here, so that data has not
        # to be read twice in case of repeating indices
        argssub = map(arg -> subsetarg(arg, chunk), bcf.args)
        view(dest, chunk...) .= bcf.f.(argssub...)
    end
    return dest
end

# DiskArrays interface

haschunks(a::BroadcastDiskArray) = Chunked()
eachchunk(a::BroadcastDiskArray) = 
    common_chunks(size(a.broadcasted), a.broadcasted.args...)
function readblock!(a::BroadcastDiskArray, aout, i::OrdinalRange...)
    argssub = map(arg -> subsetarg(arg, i), a.broadcasted.args)
    return aout .= a.broadcasted.f.(argssub...)
end

# Utility methods

function common_chunks(s, args...)
    N = length(s)
    chunkedarrays = reduce(args; init=()) do acc, x
        haschunks(x) isa Chunked ? (acc..., x) : acc
    end
    all(ar -> isa(eachchunk(ar), GridChunks), chunkedarrays) ||
        error("Currently only chunks of type GridChunks can be merged by broadcast")
    if isempty(chunkedarrays)
        totalsize = sum(sizeof ∘ eltype, args)
        return estimate_chunksize(s, totalsize)
    elseif length(chunkedarrays) == 1
        return eachchunk(only(chunkedarrays))
    else
        allchunks = collect(map(eachchunk, chunkedarrays))
        tt = ntuple(N) do n
            csnow = filter(allchunks) do cs
            ndims(cs) >= n && first(first(cs.chunks[n])) < last(last(cs.chunks[n]))
            end
            isempty(csnow) && return RegularChunks(1, 0, s[n])
            
            cs = first(csnow).chunks[n]
            if all(s -> s.chunks[n] == cs, csnow)
                return cs
            else
                return merge_chunks(csnow, n)
            end
        end
        return GridChunks(tt...)
    end
end

to_ranges(r::Tuple) = r
to_ranges(r::CartesianIndices) = r.indices

function merge_chunks(csnow, n)
    # The first offset is always zero
    offsets = Int[0]
    chpos = [1 for ch in csnow]
    while true
        # Get the largest chunk end point
        currentchunks = map(chpos, csnow) do i, ch
            ch.chunks[n][i]
        end
        chend = maximum(last.(currentchunks))
        # Find the position where the end of a chunk matches the new chunk endpoint
        newchpos = map(chpos, csnow) do i, ch
            found = findnext(x -> last(x) == chend, ch.chunks[n], i)
            found === nothing && error("Chunks do not align in dimension $n")
            found
        end
        # If we can't find this end point for all chunk lists, error
        firstcs = csnow[1].chunks[n]
        # Define the new combined chunk offset
        newchunkoffset = last(firstcs[newchpos[1]])
        # Update positions in each list of chunks
        chpos = newchpos .+ 1
        # If this is the last chunk, break
        chpos[1] > length(firstcs) && break
        # Add our new offset
        push!(offsets, newchunkoffset)
    end
    push!(offsets, last(last(csnow[1].chunks[n])))
    return IrregularChunks(offsets)
end

subsetarg(arg, ranges) = arg
# Maybe making a copy here would be faster, need to check...
subsetarg(arg::AbstractArray, ranges) = 
    view(arg, maybeonerange(size(arg), ranges)...) 

maybeonerange(sizes, ranges) = maybeonerange((), sizes, ranges)
function maybeonerange(out, sizes, ranges)
    s1, sr... = sizes
    r1, rr... = ranges
    return maybeonerange((out..., repsingle(s1, r1)), sr, rr)
end
maybeonerange(out, ::Tuple{}, ranges) = out
maybeonerange(out, sizes, ::Tuple{}) = out
maybeonerange(out, ::Tuple{}, ::Tuple{}) = out

repsingle(size, range) = size == 1 ? (1:1) : range

# Implementation macro

macro implement_broadcast(t)
    t = esc(t)
    quote
        # Broadcasting with a DiskArray on LHS
        function Base.copyto!(dest::$t, bc::Broadcasted{Nothing})
            foreach(eachchunk(dest)) do c
                ar = [bc[i] for i in CartesianIndices(to_ranges(c))]
                dest[to_ranges(c)...] = ar
            end
            return dest
        end
        Base.BroadcastStyle(T::Type{<:$t}) = ChunkStyle{ndims(T)}()
        function DiskArrays.subsetarg(arg::$t, ranges)
            ashort = maybeonerange(size(arg), ranges)
            return arg[ashort...]
        end

        # This is a heavily allocating implementation, but working for all cases.
        # As performance optimization one might:
        # Allocate the array only once if all chunks have the same size
        # Use FillArrays, if the DiskArray accepts these
        function Base.fill!(dest::$t, value)
            foreach(eachchunk(dest)) do c
                ar = fill(value, length.(to_ranges(c)))
                dest[to_ranges(c)...] = ar
            end
            return dest
        end
    end
end
