"""
    ThreadingTrait

Trait to indicate whether a DiskArray backend supports thread-safe operations.
"""
abstract type ThreadingTrait end

"""
    ThreadSafe()

Indicates that the DiskArray backend supports thread-safe read operations.
"""
struct ThreadSafe <: ThreadingTrait end

"""
    NotThreadSafe()

Indicates that the DiskArray backend does not support thread-safe operations.
Default for all backends unless explicitly overridden.
"""
struct NotThreadSafe <: ThreadingTrait end

"""
    threading_trait(::Type{T}) -> ThreadingTrait
    threading_trait(x) -> ThreadingTrait

Return the threading trait for a DiskArray type or instance.
Defaults to `NotThreadSafe()` for safety.
"""
threading_trait(::Type{<:AbstractDiskArray}) = NotThreadSafe()
threading_trait(x::AbstractDiskArray) = threading_trait(typeof(x))

"""
    is_thread_safe(x) -> Bool

Check if a DiskArray supports thread-safe operations.
"""
is_thread_safe(x) = threading_trait(x) isa ThreadSafe

"""
    AlgorithmTrait

Trait to indicate whether a method is multithreaded or not
"""
abstract type AlgorithmTrait end

"""
    SingleThreaded()

Indicates that a method uses just one thread
"""
struct SingleThreaded <: AlgorithmTrait  end

"""
    MultiThreaded()

Indicates that a method uses all threads available
"""
struct MultiThreaded <: AlgorithmTrait  end

# Global threading control
const THREADING_ENABLED = Ref(true)

"""
    enable_threading(enable::Bool=true)

Globally enable or disable threading for DiskArray operations.
When disabled, all algorithms will run single-threaded regardless of backend support.
"""
enable_threading(enable::Bool=true) = (THREADING_ENABLED[] = enable)

"""
    threading_enabled() -> Bool

Check if threading is globally enabled.
"""
threading_enabled() = THREADING_ENABLED[]

"""
    should_use_threading(x) -> Val(Bool)

Determine if threading should be used for a given DiskArray.
Returns true only if both global threading is enabled AND the backend is thread-safe.
"""
should_use_threading(x) = threading_enabled() && is_thread_safe(x) ? MultiThreaded : SingleThreaded
