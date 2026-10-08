"""
    DiskArraysCore

The interface and chunk types of [DiskArrays.jl](https://github.com/JuliaIO/DiskArrays.jl),
without any of its implementations.

This package defines [`AbstractDiskArray`](@ref), the functions that disk array types
implement (`readblock!`, `writeblock!`, `eachchunk`, `haschunks`, `chunkexists`), and the
chunk types (`RegularChunks`, `IrregularChunks`, `GridChunks`, `Chunked`, `Unchunked`,
`ChunkIndex`, `ChunkIndices`). It has no dependencies and does not add methods to
`Base` functions that are shared with other array types (such as `Base.Generator` or `zip`),
so loading it does not invalidate compiled code.

The generic indexing, broadcasting, reduction, etc. methods for `AbstractDiskArray` are
defined by DiskArrays.jl, which re-exports everything here.
"""
module DiskArraysCore

export AbstractDiskArray, eachchunk, chunkexists, ChunkIndex, ChunkIndices

include("batchstrategy.jl")
include("chunks.jl")
include("diskarray.jl")

end # module
