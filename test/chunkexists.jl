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

    @testset "ConcatDiskArray with missing tiles" begin
        # Tiles are 4x4 with 2x2 chunks, so the concat array has a 4x4 chunk grid
        # and every chunk maps to a single tile.
        tile() = AccessCountDiskArray(zeros(4, 4); chunksize=(2, 2))
        missingtile = DiskArrays.MissingTile(0.0)
        tiles = reshape([tile(), missingtile, missingtile, tile()], 2, 2)
        a = DiskArrays.ConcatDiskArray(tiles)
        @test size(eachchunk(a)) == (4, 4)
        @test chunkexists(a, 1, 1)
        @test chunkexists(a, 2, 2)
        @test !chunkexists(a, (1, 3))
        @test !chunkexists(a, CartesianIndex(3, 1))
        @test chunkexists(a, ChunkIndex(4, 4))
        @test chunkexists(a, ChunkIndex(1, 3; offset=true)) == false
        indices = [(4, 4), (1, 4), (4, 4)]
        @test chunkexists(a, indices) == [true, false, true]
        @test chunkexists(a, (i for i in indices if i[1] == 4)) == [true, true]
        @test chunkexists(a, NTuple{2,Int}[]) == Bool[]
        @test eltype(chunkexists(a, (i for i in indices if false))) == Bool
        @test chunkexists(a, Tuple(indices)) == [true, false, true]
        expected = [i <= 2 && j <= 2 || i > 2 && j > 2 for i in 1:4, j in 1:4]
        @test chunkexists(a, CartesianIndices((4, 4))) == expected
        @test chunkexists(a, ChunkIndices(a)) == expected
        @test all(getindex_count(t) == 0 for t in tiles if t isa AccessCountDiskArray)

        # Stacking along a new dimension: the chunk index in the new dimension selects the tile
        v = DiskArrays.ConcatDiskArray(reshape([tile(), missingtile, tile()], 1, 1, 3))
        @test size(eachchunk(v)) == (2, 2, 3)
        @test chunkexists(v, [(1, 1, 1), (2, 2, 2), (1, 2, 3)]) == [true, false, true]
        @test chunkexists(v, ChunkIndices(v)) == cat(trues(2, 2), falses(2, 2), trues(2, 2); dims=3)

        # Queries are forwarded to the tiles' own chunkexists
        nested = DiskArrays.ConcatDiskArray(reshape([a, missingtile, missingtile, a], 2, 2))
        @test size(eachchunk(nested)) == (8, 8)
        @test chunkexists(nested, ChunkIndices(nested)) == [expected falses(4, 4); falses(4, 4) expected]
    end
end
