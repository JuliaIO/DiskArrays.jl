import Mmap

"""
    ChunkTiledDiskArray <: AbstractDiskArray

And abstract supertype for disk arrays that have fast indexing
of tiled chunks already stored as separate arrays, such as [`CachedDiskArray`](@ref).
"""
abstract type ChunkTiledDiskArray{T,N} <: AbstractDiskArray{T,N} end

Base.size(a::ChunkTiledDiskArray) = arraysize_from_chunksize.(eachchunk(a).chunks)

function readblock!(A::ChunkTiledDiskArray{T,N}, data, I...) where {T,N}
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
    CachedDiskArray <: ChunkTiledDiskArray

    CachedDiskArray(A::AbstractArray; maxsize=1000, mmap=false)

Wrap some disk array `A` with a caching mechanism that will 
keep chunks up to a total of `maxsize` megabytes, dropping
the least used chunks when `maxsize` is exceeded. If `mmap` is
set to `true`, cached chunks will not be kept in RAM but Mmapped 
to temproray files.  

Can also be called with `cache`, which can be extended for wrapper array types.
"""
struct CachedDiskArray{T,N,A<:AbstractArray{T,N},C} <: ChunkTiledDiskArray{T,N}
    parent::A
    cache::C
    mmap::Bool
end
function CachedDiskArray(A::AbstractArray{T,N}; maxsize=1000, mmap=false) where {T,N}
    by(x) = sizeof(x) ÷ 1_000_000 # In Megabytes
    CachedDiskArray(A, LRU{ChunkIndex{N,OffsetChunks},OffsetArray{T,N,Array{T,N}}}(; by, maxsize),mmap)
end

# Scalar indexing is allowed on CachedDiskArray
checkscalar(::Type{Bool}, a::CachedDiskArray, i::Tuple) = true
checkscalar(::Type{Bool}, a::CachedDiskArray, i::Tuple{}) = true

Base.parent(A::CachedDiskArray) = A.parent
Base.size(A::CachedDiskArray) = size(parent(A))
# TODO we need to invalidate caches when we write
# writeblock!(A::CachedDiskArray, data, I...) = writeblock!(parent(A), data, I...)

haschunks(A::CachedDiskArray) = haschunks(parent(A))
eachchunk(A::CachedDiskArray) = eachchunk(parent(A))

# OffsetChunks return an OffsetArray, OneBasedChunks an Array
Base.getindex(A::CachedDiskArray, i::ChunkIndex{N,OffsetChunks}) where {N} = 
    _getchunk(A, i)
Base.getindex(A::CachedDiskArray, i::ChunkIndex{N,OneBasedChunks}) where {N} = 
    parent(_getchunk(A, i))

function _getchunk(A::CachedDiskArray, i::ChunkIndex)
    get!(A.cache, i) do
        inds = eachchunk(A)[i.I]
        chunk = parent(A)[inds...]
        if A.mmap
            mmappedarray = Mmap.mmap(
                tempname(), 
                Array{eltype(chunk),ndims(chunk)}, 
                size(chunk); 
                shared=false
            )
            copyto!(mmappedarray, chunk)
            chunk = mmappedarray
        end
        wrapchunk(chunk, inds)
    end
end


"""
    cache(A::AbstractArray; maxsize=1000, mmap=false)

Wrap internal disk arrays with `CacheDiskArray`.

This function is intended to be extended by package that want to
re-wrap the disk array afterwards, such as YAXArrays.jl or Rasters.jl.
"""
cache(A::AbstractArray; maxsize=1000, mmap=false) = CachedDiskArray(A; maxsize, mmap)

