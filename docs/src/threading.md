# Threading Support

DiskArrays.jl provides support for threaded algorithms when the underlying
storage backend supports thread-safe read operations.

## Threading Trait System

The threading support is based on a trait system that allows backends to
declare whether they support thread-safe operations:

```julia
using DiskArrays

# Check if an array supports threading
is_thread_safe(my_array)

# Get the threading trait
threading_trait(my_array)  # Returns ThreadSafe() or NotThreadSafe()
```

## Global Threading Control

You can globally enable or disable threading for all DiskArray operations:

```julia
# Disable threading globally
disable_threading()

# Enable threading globally (default)
enable_threading()

# Check current status
threading_enabled()
```

## Implementing Threading Support in Backends

Backend developers can opt into threading support by overriding the threading_trait method:

```julia
# For a hypothetical ThreadSafeArray type
DiskArrays.threading_trait(::Type{ThreadSafeArray}) = DiskArrays.ThreadSafe()
```

Important: Only declare your backend as thread-safe if:

* Multiple threads can safely read from the storage simultaneously
* The underlying storage system (files, network, etc.) supports concurrent access
* No global state is modified during read operations

## Implementing Threading Support for Disk Array Methods

Add a (or rename the existing) single-threaded method using this signature:

```
function Base.myfun(::Type{SingleThreaded}, ...)
```

Write a threaded version using this signature:

```
function Base.myfun(::Type{MultiThreaded}, ...)
```

Add this additional method to automatically dispatch between the two:

```
Base.myfun(v::AbstractDiskArray, ...) = myfun(should_use_threading(v), ...)
```

## Threaded Algorithms

Currently supported threaded algorithms:

### unique

```julia
# Will automatically use threading if backend supports it
result = unique(my_disk_array)

# With a function
result = unique(x -> x % 10, my_disk_array)

# Explicitly use threaded version
result = unique(MultiThreaded, f, my_disk_array)
```

The threaded unique algorithm:

* Processes each chunk in parallel using `Threads.@threads`
* Combines results using a reduction operation
* Falls back to single-threaded implementation for non-thread-safe backends

## Performance Considerations

* Threading is most beneficial for arrays with many chunks
* I/O bound operations may see limited speedup due to storage bottlenecks
* Consider the overhead of thread coordination for small arrays
* Test with your specific storage backend and access patterns
