# Backend System

## Overview

Computations on `AbstractDiskArray` objects use a backend dispatch mechanism. Each reduction
function (e.g. `sum`, `mean`, `mapreduce`) calls a `diskarrays_<func>_impl(f, a, backend)`
dispatch function, which defaults to a single-threaded chunked iterator but can be overridden
for any `ComputeBackend` subtype.

The global backend is selected at module load time via the `"backend"` preference and stored
in `const compute_backend`. It is wrapped in a `DynamicBackend` which allows runtime switching
via `set_dynamic_backend!` or `set_backend`. All entry points call `get_backend(a)`, which walks
up the parent chain of `a` looking for a `WithBackendDiskArray` wrapper (created by `withbackend`),
then falls back to the global backend.

## Built-in Types

| Type | Description |
|------|-------------|
| `ComputeBackend` | Abstract supertype for all backends |
| `DefaultBackend` | Single-threaded chunked iterator |
| `DiskArrayEngineBackend` | Used when DiskArrayEngine.jl is loaded |

## Extendable Functions

The following `diskarrays_*_impl` functions are the extension points. Each has a default
implementation for `::ComputeBackend` and can be specialized for any `ComputeBackend` subtype.

All functions accept a function `f` as the first argument, the disk array as the second, and a
`ComputeBackend` as the third. Additional keyword arguments are passed through.

| Function | Signature |
|----------|-----------|
| `diskarrays_sum_impl` | `diskarrays_sum_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_prod_impl` | `diskarrays_prod_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_all_impl` | `diskarrays_all_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_any_impl` | `diskarrays_any_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_minimum_impl` | `diskarrays_minimum_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_maximum_impl` | `diskarrays_maximum_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_mapreduce_impl` | `diskarrays_mapreduce_impl(f, op, a, dims, init, backend::ComputeBackend)` |
| `diskarrays_mapreducedim_impl` | `diskarrays_mapreducedim_impl(f, op, R, a::AbstractDiskArray, backend::ComputeBackend)` |
| `diskarrays_count_impl` | `diskarrays_count_impl(f, v::AbstractDiskArray, ::ComputeBackend)` |
| `diskarrays_unique_impl` | `diskarrays_unique_impl(f, v::AbstractDiskArray, ::ComputeBackend)` |
| `diskarrays_extrema_impl` | `diskarrays_extrema_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_mean_impl` | `diskarrays_mean_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_median_impl` | `diskarrays_median_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |

## Per-array Backend Override

`withbackend(a, backend)` wraps an array so that `get_backend(a)` returns the specified backend
instead of the global one. This is mainly useful for testing or for mixing arrays with
different backends in the same expression.

```julia
a = withbackend(my_diskarray, DiskArrayEngineBackend())
sum(a)  # uses DiskArrayEngineBackend regardless of the global setting
```

Note that `BroadcastStyle` uses the global backend — it only sees the array's type, not the
`withbackend` wrapper.

## Error Hints

DiskArrays registers a `MethodError` hint for `DiskArrayEngineBackend`. When a method is not
found, it prints:

- `DiskArrayEngine.jl does not implement this operation for DiskArrayEngineBackend.` (if
  DiskArrayEngine is already loaded)
- `DiskArrayEngineBackend methods are implemented in DiskArrayEngine.jl, make sure it is loaded
  with using DiskArrayEngine.` (if DiskArrayEngine is not yet loaded)
