module DiskArrays

import ConstructionBase
import Base.PermutedDimsArrays: genperm

using LRUCache: LRUCache, LRU

using Base: tail

# The interface (`AbstractDiskArray`, `readblock!`, ...) and the chunk types live in
# DiskArraysCore, which has no dependencies and does not invalidate any compiled code.
# DiskArrays adds the generic implementations of the array methods.
using DiskArraysCore: DiskArraysCore
import DiskArraysCore:
    AbstractDiskArray, AllowStepRange, BatchStrategy, CanStepRange, ChunkRead, NoBatch,
    NoStepRange, SubRanges, ChunkIndex, ChunkIndexType, ChunkIndices, ChunkVector, Chunked,
    ChunkedTrait, GridChunks, IrregularChunks, OffsetChunks, OneBasedChunks, RegularChunks,
    Unchunked, approx_chunksize, arraysize_from_chunksize, batchstrategy, chunk_runlength,
    chunkexists, chunktype_from_chunksizes, default_chunk_size, eachchunk, element_size,
    estimate_chunksize, fallback_element_size, findchunk, grid_offset, haschunks, isdisk,
    max_chunksize, nooffset, readblock!, subsetchunks, subsetchunks_fallback, writeblock!

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DiskArrays

using OffsetArrays: OffsetArray

# Used by `OffsetChunks` indices to return a chunk with its position in the array
wrapchunk(x, inds) = OffsetArray(x, inds...)

export AbstractDiskArray, eachchunk, chunkexists, ChunkIndex, ChunkIndices, MissingTile

include("scalar.jl")
include("batchgetindex.jl")
include("diskindex.jl")
include("indexing.jl")
include("rangeindex.jl")
include("array.jl")
include("broadcast.jl")
include("iterator.jl")
include("mapreduce.jl")
include("permute.jl")
include("reshape.jl")
include("subarray.jl")
include("mockchunks.jl")
include("cat.jl")
include("generator.jl")
include("zip.jl")
include("show.jl")
include("chunktiledarray.jl")
include("cached.jl")
include("pad.jl")

# The all-in-one macro

macro implement_diskarray(t)
    # Need to do this for dispatch ambiguity
    t = esc(t)
    quote
        @implement_getindex $t
        @implement_setindex $t
        @implement_broadcast $t
        @implement_iteration $t
        @implement_mapreduce $t
        @implement_reshape $t
        @implement_array_methods $t
        @implement_permutedims $t
        @implement_subarray $t
        @implement_cat $t
        @implement_zip $t
        @implement_show $t
        @implement_generator $t
    end
end

# https://github.com/JuliaIO/DiskArrays.jl/issues/175
macro implement_diskarray_skip_zip(t)
    # Need to do this for dispatch ambiguity
    t = esc(t)
    quote
        @implement_getindex $t
        @implement_setindex $t
        @implement_broadcast $t
        @implement_iteration $t
        @implement_mapreduce $t
        @implement_reshape $t
        @implement_array_methods $t
        @implement_permutedims $t
        @implement_subarray $t
        @implement_cat $t
        @implement_show $t
        @implement_generator $t
    end
end

# We need to skip the `implement_zip` macro for dispatch
@implement_diskarray_skip_zip AbstractDiskArray

# And we define the test types
include("util/testtypes.jl")


end # module
