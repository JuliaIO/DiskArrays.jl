struct ChunkExistenceArray{N} <: AbstractDiskArray{Int,N}
    stored::Set{NTuple{N,Int}}
    queries::Vector{NTuple{N,Int}}
end
Base.size(::ChunkExistenceArray{N}) where {N} = ntuple(_ -> 4, N)
DiskArrays.eachchunk(a::ChunkExistenceArray{N}) where {N} =
    DiskArrays.GridChunks(a, ntuple(_ -> 2, N))
function DiskArrays.chunkexists(a::ChunkExistenceArray, idxs::Integer...)
    push!(a.queries, idxs)
    return idxs in a.stored
end

struct BatchChunkExistenceArray{N} <: AbstractDiskArray{Int,N}
    parent::ChunkExistenceArray{N}
    queries::Vector{Vector{NTuple{N,Int}}}
end
DiskArrays.chunkexists(a::BatchChunkExistenceArray, idxs::Integer...) =
    chunkexists(a.parent, idxs...)
function DiskArrays.chunkexists(a::BatchChunkExistenceArray, indices::AbstractVector{<:Tuple})
    push!(a.queries, collect(indices))
    return [i in a.parent.stored for i in indices]
end

@testset "chunkexists" begin
    @testset "default" begin
        a = AccessCountDiskArray(zeros(4, 4); chunksize=(2, 2))
        @test chunkexists(a, 1, 2)
        @test chunkexists(a, (1, 2))
        @test chunkexists(a, CartesianIndex(1, 2))
        @test chunkexists(a, ChunkIndex(1, 2; offset=true))
        @test chunkexists(a, [(1, 1), (2, 2)]) == [true, true]
        @test chunkexists(a, ChunkIndices(a)) == trues(2, 2)
        @test getindex_count(a) == 0
        @test chunkexists(zeros(4), 1)
        @test chunkexists(fill(0))
        @test chunkexists(fill(0), ())
        @test chunkexists(fill(0), [()]) == [true]
    end

    @testset "scalar override and batch fallback" begin
        a = ChunkExistenceArray(Set([(1, 1), (2, 2)]), NTuple{2,Int}[])
        @test chunkexists(a, 1, 1)
        @test !chunkexists(a, (1, 2))
        @test !chunkexists(a, CartesianIndex(2, 1))
        @test chunkexists(a, ChunkIndex(2, 2))
        indices = [(2, 2), (1, 2), (2, 2)]
        empty!(a.queries)
        @test chunkexists(a, indices) == [true, false, true]
        @test a.queries == indices
        empty!(a.queries)
        @test chunkexists(a, (i for i in indices if i[1] == 2)) == [true, true]
        @test a.queries == [(2, 2), (2, 2)]
        empty!(a.queries)
        @test chunkexists(a, NTuple{2,Int}[]) == Bool[]
        @test eltype(chunkexists(a, (i for i in indices if false))) == Bool
        @test isempty(a.queries)
        @test chunkexists(a, Tuple(indices)) == [true, false, true]
        @test chunkexists(a, CartesianIndices((2, 2))) == [true false; false true]
        @test chunkexists(a, ChunkIndices(a)) == [true false; false true]
        @test chunkexists(a, ChunkIndex(1, 2; offset=true)) == false
        v = ChunkExistenceArray(Set([(2,)]), NTuple{1,Int}[])
        @test chunkexists(v, [1, 2]) == [false, true]
    end

    @testset "batch override" begin
        parent = ChunkExistenceArray(Set([(1, 1)]), NTuple{2,Int}[])
        a = BatchChunkExistenceArray(parent, Vector{NTuple{2,Int}}[])
        indices = [(1, 1), (1, 2), (1, 1)]
        @test chunkexists(a, indices) == [true, false, true]
        @test a.queries == [indices]
        @test isempty(parent.queries)
        @test chunkexists(a, 1, 1)
        @test parent.queries == [(1, 1)]
    end
end
