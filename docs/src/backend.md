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
`ComputeBackend` as the third.

| Function | Signature |
|----------|-----------|
| `diskarrays_sum_impl` | `diskarrays_sum_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:, kwargs...)` |
| `diskarrays_prod_impl` | `diskarrays_prod_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:, kwargs...)` |
| `diskarrays_all_impl` | `diskarrays_all_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:)` |
| `diskarrays_any_impl` | `diskarrays_any_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:)` |
| `diskarrays_minimum_impl` | `diskarrays_minimum_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:, kwargs...)` |
| `diskarrays_maximum_impl` | `diskarrays_maximum_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:, kwargs...)` |
| `diskarrays_mapreduce_impl` | `diskarrays_mapreduce_impl(f, op, a, dims, init, backend::ComputeBackend)` |
| `diskarrays_mapreducedim_impl` | `diskarrays_mapreducedim_impl(f, op, R, a::AbstractDiskArray, backend::ComputeBackend)` |
| `diskarrays_count_impl` | `diskarrays_count_impl(f, v::AbstractDiskArray, ::ComputeBackend; dims=:, init=0)` |
| `diskarrays_unique_impl` | `diskarrays_unique_impl(f, v::AbstractDiskArray, ::ComputeBackend)` |
| `diskarrays_extrema_impl` | `diskarrays_extrema_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:, kwargs...)` |
| `diskarrays_mean_impl` | `diskarrays_mean_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |
| `diskarrays_median_impl` | `diskarrays_median_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)` |

Each reduction reaches its own hook from every call form with a `Function`, with or without
`dims` and `init`, as listed in the table below. The hook receives only the keywords the caller
passed, so a backend that implements whole-array reductions without `init` alone can define it
without keywords. The defaults reduce whole arrays chunk by chunk, and pass `dims` and `init`
on to `mapreduce`, which reaches `diskarrays_mapreduce_impl`.

| call | hook |
|------|------|
| `sum(a)`, `sum(f::Function, a)` | `diskarrays_sum_impl(f, a, backend)` |
| `sum(f, a; init)`, `sum(a; init)` | `diskarrays_sum_impl(f, a, backend; init)` |
| `sum(f, a; dims)`, `sum(f, a; dims, init)` | `diskarrays_sum_impl(f, a, backend; dims, init)` |
| `sum(f, a)` for a callable struct `f` | `diskarrays_mapreduce_impl(f, Base.add_sum, a, :, Base._InitialValue(), backend)` |
| `count(f, a; dims, init)`, every form | `diskarrays_count_impl(f, a, backend; dims, init)` |
| `any(f, a; dims)`, every form | `diskarrays_any_impl(f, a, backend; dims)` |

`prod`, `minimum`, `maximum` and `extrema` follow `sum`, and `all` follows `any`.

DiskArrays reaches these hooks through Base's internal entry points where it can, which keeps
compiled code in other packages valid when DiskArrays loads. Every public `sum` method with
`dims` or `init` calls `Base._sum(f, a, dims; kw...)`, and DiskArrays adds a method for
`a::AbstractDiskArray` there. For whole-array calls with `init` it is only the keyword-call
method of `Base._sum(f, a, ::Colon)`. That method invalidates compiled `maximum(f, v; init)`
calls in other packages, such as GeometryBasics, which costs 6–13 MethodInstances in a
Makie session. For `any` and `all` the entry point depends on the Julia version:

| Julia | `any(f, a)`, `all(f, a)` | with `dims` |
|-------|--------------------------|-------------|
| 1.10 | public `any(f::Function, a)`; other callables iterate | `Base._any(f, a, dims)` |
| 1.11 and later | `Base._any(f, a, ::Colon)`, any callable | `Base._any(f, a, dims)` |

The default `any`/`all` hooks read chunk by chunk and stop at the first chunk that decides the
result.

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

## Broadcasting

| Function | Signature |
|----------|-----------|
| `diskarrays_broadcaststyle` | `diskarrays_broadcaststyle(T::Type, ::ComputeBackend)`, defaults to `ChunkStyle{ndims(T)}()` |
| `diskarrays_coptyo!` | `diskarrays_coptyo!(dest, bc::Broadcasted, ::ComputeBackend)`, for a `Broadcasted{Nothing}` into a disk array |
| `diskarrays_fill!` | `diskarrays_fill!(dest, value, ::ComputeBackend)` |

Broadcasting over a disk array is lazy: `s = a .+ 1` is a `BroadcastDiskArray`. When `s` is
used in another broadcast, `s .* b`, DiskArrays fuses both into one pass over `a` and `b` by
calling `DiskArrays.unwrap_broadcast` before `Broadcast.flatten`. A backend that returns its own
style from `diskarrays_broadcaststyle` and defines `copy`/`copyto!` for it should do the same:

```julia
Base.copy(bc::Broadcasted{MyStyle{N}}) where {N} = my_lazy_array(Broadcast.flatten(DiskArrays.unwrap_broadcast(bc)))
```

`diskarrays_coptyo!` already receives the unwrapped expression.

## Error Hints

DiskArrays registers a `MethodError` hint for `DiskArrayEngineBackend`. When a method is not
found, it prints:

- `DiskArrayEngine.jl does not implement this operation for DiskArrayEngineBackend.` (if
  DiskArrayEngine is already loaded)
- `DiskArrayEngineBackend methods are implemented in DiskArrayEngine.jl, make sure it is loaded
  with using DiskArrayEngine.` (if DiskArrayEngine is not yet loaded)
