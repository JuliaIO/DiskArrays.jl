using Test
using DiskArraysCore
using DiskArraysCore: GridChunks, RegularChunks, IrregularChunks, Chunked, Unchunked,
    haschunks, readblock!, writeblock!, estimate_chunksize

# A minimal disk array that only implements the interface
struct MyDisk{T,N} <: AbstractDiskArray{T,N}
    data::Array{T,N}
end
Base.size(a::MyDisk) = size(a.data)
DiskArraysCore.readblock!(a::MyDisk, aout, r::AbstractUnitRange...) =
    (aout .= view(a.data, r...); nothing)
DiskArraysCore.haschunks(::MyDisk) = Chunked()
DiskArraysCore.eachchunk(a::MyDisk) = GridChunks(size(a), (2, 3))

@testset "interface" begin
    a = MyDisk(collect(reshape(1:12, 3, 4)))
    @test a isa AbstractArray{Int,2}
    @test haschunks(a) isa Chunked
    @test haschunks([1, 2, 3]) isa Unchunked
    @test vec(collect(eachchunk(a))) == [(1:2, 1:3), (3:3, 1:3), (1:2, 4:4), (3:3, 4:4)]
    out = zeros(Int, 2, 3)
    readblock!(a, out, 1:2, 1:3)
    @test out == a.data[1:2, 1:3]
    @test chunkexists(a, 1, 1)
    @test eachchunk([1.0 2; 3 4]) isa GridChunks
end

@testset "chunks" begin
    @test RegularChunks(3, 0, 10) isa AbstractVector
    @test length(RegularChunks(3, 0, 10)) == 4
    @test IrregularChunks(; chunksizes=[2, 3, 5]) == [1:2, 3:5, 6:10]
    ci = ChunkIndex(1, 2)
    @test ci isa ChunkIndex{2}
    ids = ChunkIndices((1:2, 1:3), DiskArraysCore.OneBasedChunks())
    @test size(ids) == (2, 3)
    @test eltype(ids) === ChunkIndex{2,DiskArraysCore.OneBasedChunks}
end

@testset "no invalidating Base methods on shared types" begin
    for f in (Base.Generator, zip, Base.collect, Base.map)
        @test !any(m -> m.module === DiskArraysCore, methods(f))
    end
end
