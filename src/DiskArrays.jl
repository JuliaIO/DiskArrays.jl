module DiskArrays

import ConstructionBase
import Base.PermutedDimsArrays: genperm

using LRUCache: LRUCache, LRU
using OffsetArrays: OffsetArray

using Base: tail

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DiskArrays

export AbstractDiskArray, eachchunk, chunkexists, ChunkIndex, ChunkIndices, backend, withbackend,MissingTile

include("scalar.jl")
include("chunks.jl")
include("compute.jl")
include("diskarray.jl")
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
include("show.jl")
include("chunktiledarray.jl")
include("cached.jl")
include("pad.jl")
include("withbackend.jl")

# The all-in-one macro

macro implement_diskarray(t)
    # Need to do this for dispatch ambiguity
    t = esc(t)
    quote
        @implement_getindex $t
        @implement_setindex $t
        @implement_broadcast $t
        @implement_iteration $t
        @implement_reshape $t
        @implement_array_methods $t
        @implement_permutedims $t
        @implement_subarray $t
        @implement_cat $t
        @implement_show $t
    end
end

@implement_diskarray AbstractDiskArray

# And we define the test types
include("util/testtypes.jl")

function __init__()
    Base.Experimental.register_error_hint(_backend_error_hint, MethodError)
end


end # module
