"""
    AbstractChunkTiledDiskArray <: AbstractDiskArray

An abstract supertype for disk arrays that have fast indexing
of tiled chunks already stored as separate arrays, such as [`CachedDiskArray`](@ref).
"""
abstract type AbstractChunkTiledDiskArray{T,N} <: AbstractDiskArray{T,N} end

Base.size(a::AbstractChunkTiledDiskArray) = arraysize_from_chunksize.(eachchunk(a).chunks)

function readblock!(A::AbstractAbstractChunkTiledDiskArray{T,N}, data, I...) where {T,N}
    chunks = eachchunk(A)
    chunk_indices = findchunk.(chunks.chunks, I)
    data_offset = OffsetArray(data, map(i -> first(i) - 1, I)...)
    foreach(CartesianIndices(chunk_indices)) do ci
        chunkindex = ChunkIndex(ci; offset=true)
        chunk = A[chunkindex]
        # Find the overlapping indices
        inner_indices = map(axes(chunk), axes(data_offset)) do ax1, ax2
            max(first(ax1), first(ax2)):min(last(ax1), last(ax2))
        end
        for ii in CartesianIndices(inner_indices)
            data_offset[ii] = chunk[ii]
        end
    end
end

"""
    TiledDiskArray <: AbstractChunkTiledDiskArray

Construct an array from a collection of tiles. 
This needs a function to find the tile given a tile position and the overall size of the array.
"""
struct TiledDiskArray{T,N} <: AbstractChunkTiledDiskArray{T,N}
    tilefunction
    size::NTuple{N, Int}
end
