"""
    AbstractChunkTiledDiskArray <: AbstractDiskArray

An abstract supertype for disk arrays that have fast indexing
of tiled chunks already stored as separate arrays, such as [`CachedDiskArray`](@ref).
"""
abstract type AbstractChunkTiledDiskArray{T,N} <: AbstractDiskArray{T,N} end

Base.size(a::AbstractChunkTiledDiskArray) = arraysize_from_chunksize.(eachchunk(a).chunks)

function readblock!(A::AbstractChunkTiledDiskArray{T,N}, data, I...) where {T,N}
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
struct TiledDiskArray{T,N,F,G<:GridChunks{N}} <: AbstractChunkTiledDiskArray{T,N}
    tilefunction::F
    tileshape::G
end
export TiledDiskArray
TiledDiskArray(f,T,tilenum, tilesize) = TiledDiskArray{T,length(tilenum),typeof(f),typeof(GridChunks(tilenum.*tilesize, tilesize))}(f,GridChunks(tilenum.*tilesize, tilesize))

Base.size(A::TiledDiskArray) = map(arraysize_from_chunksize,A.tileshape.chunks)
eachchunk(A::TiledDiskArray) = A.tileshape
haschunks(::TiledDiskArray) = Chunked()

function Base.getindex(A::TiledDiskArray, i::ChunkIndex{N,OffsetChunks}) where {N}
    tile = _getchunk(A,i)
    inds = eachchunk(A)[i.I]
    wrapchunk(tile, inds)
end

Base.getindex(A::TiledDiskArray, i::ChunkIndex{N,OneBasedChunks}) where {N} = 
    _getchunk(A, i)



function _getchunk(A::TiledDiskArray, i::ChunkIndex)
    A.tilefunction(i.I.I...)
end
