# Mock thread-safe DiskArray for testing
struct MockThreadSafeDiskArray{T,N} <: AbstractDiskArray{T,N}
    data::Array{T,N}
    chunks::NTuple{N,Int}
end

Base.size(a::MockThreadSafeDiskArray) = size(a.data)
Base.getindex(a::MockThreadSafeDiskArray, i::Int...) = a.data[i...]
DiskArrays.eachchunk(a::MockThreadSafeDiskArray) = DiskArrays.GridChunks(a, a.chunks)
DiskArrays.haschunks(::MockThreadSafeDiskArray) = DiskArrays.Chunked()
DiskArrays.readblock!(a::MockThreadSafeDiskArray, aout, r::AbstractUnitRange...) = (aout .= a.data[r...])

# Override threading trait for our mock array
DiskArrays.threading_trait(::Type{<:MockThreadSafeDiskArray}) = DiskArrays.ThreadSafe()

@testset "Threading Traits" begin
    # Test default behavior (not thread safe)
    regular_array = ChunkedDiskArray(rand(10, 10), (5, 5))
    @test DiskArrays.threading_trait(regular_array) isa DiskArrays.NotThreadSafe
    @test !DiskArrays.is_thread_safe(regular_array)

    # Test thread-safe array
    thread_safe_array = MockThreadSafeDiskArray(rand(10, 10), (5, 5))
    @test DiskArrays.threading_trait(thread_safe_array) isa DiskArrays.ThreadSafe
    @test DiskArrays.is_thread_safe(thread_safe_array)
end

@testset "Threading Control" begin
    # Test global threading control
    @test DiskArrays.threading_enabled()  # Should be true by default

    DiskArrays.enable_threading(false)
    @test !DiskArrays.threading_enabled()

    DiskArrays.enable_threading()
    @test DiskArrays.threading_enabled()

    # Test should_use_threading logic
    thread_safe_array = MockThreadSafeDiskArray(rand(10, 10), (5, 5))
    regular_array = ChunkedDiskArray(rand(10, 10), (5, 5))

    DiskArrays.enable_threading()
    @test DiskArrays.should_use_threading(thread_safe_array) == MultiThreaded
    @test DiskArrays.should_use_threading(regular_array) == SingleThreaded

    DiskArrays.enable_threading(false)
    @test DiskArrays.should_use_threading(thread_safe_array) == SingleThreaded
    @test DiskArrays.should_use_threading(regular_array) == SingleThreaded

    # Reset to default
    DiskArrays.enable_threading()
end

@testset "Threaded unique" begin
    # Test with thread-safe array
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7]
    reshape_data = reshape(data, 4, 5)
    thread_safe_array = MockThreadSafeDiskArray(reshape_data, (2, 3))

    result = unique(thread_safe_array)
    expected = unique(data)
    @test sort(result) == sort(expected)

    # Test with function
    result_with_func = unique(x -> x % 3, thread_safe_array)
    expected_with_func = unique(x -> x % 3, data)
    @test sort(result_with_func) == sort(expected_with_func)

    # Test fallback for non-thread-safe array
    regular_array = ChunkedDiskArray(reshape_data, (2, 3))
    result_fallback = unique(regular_array)
    @test sort(result_fallback) == sort(expected)

    # Test with threading disabled
    DiskArrays.enable_threading(false)
    result_no_threading = unique(thread_safe_array)
    @test sort(result_no_threading) == sort(expected)
    DiskArrays.enable_threading()  # Reset
end
