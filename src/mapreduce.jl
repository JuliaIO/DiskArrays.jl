
# Implementation macro

function Base._mapreduce(f, op, ::IndexCartesian, v::AbstractDiskArray)
    mapreduce(op, eachchunk(v)) do cI
        a = v[to_ranges(cI)...]
        mapreduce(f, op, a)
    end
end
function Base.mapreducedim!(f, op, R::AbstractArray, a::AbstractDiskArray)
    diskarrays_mapreducedim_impl(f, op, R, a, get_backend(a))
end

function diskarrays_mapreducedim_impl(f, op, R, a::AbstractDiskArray, ::ComputeBackend)
    _diskarrays_mapreducedim_default!(f, op, R, a)
end

function _diskarrays_mapreducedim_default!(f, op, R, a::AbstractDiskArray)
    foreach(eachchunk(a)) do cI
        aview = a[to_ranges(cI)...]
        ainds = map(
            (cinds, arsize) -> arsize == 1 ? Base.OneTo(1) : cinds,
            to_ranges(cI),
            size(R),
        )
        Base.mapreducedim!(f, op, view(R, ainds...), aview)
    end
    return R
end

function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::AbstractDiskArray)
    cc = eachchunk(itr)
    isempty(cc) &&
        return Base.mapreduce_empty_iter(f, op, itr, Base.IteratorEltype(itr))
    return Base.mapfoldl_impl(f, op, nt, itr, cc)
end
function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::AbstractDiskArray, cc)
    y = first(cc)
    a = itr[to_ranges(y)...]
    init = mapfoldl(f, op, a)
    return Base.mapfoldl_impl(f, op, (init=init,), itr, Iterators.drop(cc, 1))
end
function Base.mapfoldl_impl(f, op, nt::NamedTuple{(:init,)}, itr::AbstractDiskArray, cc)
    init = nt.init
    for y in cc
        a = itr[to_ranges(y)...]
        init = mapfoldl(f, op, a; init=init)
    end
    return init
end

Base.mapreduce(f, op, a::AbstractDiskArray; dims=:, init=Base._InitialValue(), kwargs...) =
    diskarrays_mapreduce_impl(f, op, a, dims, init, get_backend(a); kwargs...)

diskarrays_mapreduce_impl(f, op, a, dims, init, backend::ComputeBackend) =
    _diskarrays_mapreduce_impl(f, op, a, dims, init, backend)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, ::Colon, ::Base._InitialValue, ::ComputeBackend) =
    Base.mapfoldl(f, op, a)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, ::Colon, init, ::ComputeBackend) =
    Base.mapfoldl_impl(f, op, init, a)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, dims, init, ::ComputeBackend) =
    Base.mapreducedim!(f, op, fill(init, Base.reduced_indices(a, dims)), a)

_diskarrays_mapreduce_impl(f, op, a::AbstractDiskArray, dims, ::Base._InitialValue, ::ComputeBackend) =
    Base.mapreducedim!(f, op, Base.reducedim_init(f, op, a, dims), a)



# ── Backend hooks for sum, prod, minimum, maximum, extrema ──────────
# Whole-array calls enter through public methods without keywords. Any method that takes a
# whole-array `init` adds a `Core.kwcall` method that invalidates compiled
# `maximum(f, ::AbstractVector; init)` callers (GeometryBasics, so every Makie session); those
# calls reach the backend through `mapreduce`.
for fname in (:sum, :prod, :minimum, :maximum, :extrema)
    @eval begin
        # `F` forces specialization on `f`, which is only passed through here
        Base.$fname(f::F, a::AbstractDiskArray) where {F<:Function} =
            $(Symbol(:diskarrays_, fname, :_impl))(f, a, get_backend(a))
        Base.$fname(a::AbstractDiskArray) = Base.$fname(identity, a)
    end
end

# Every public `sum` method calls Base's internal `_sum(f, a, dims; kw...)`, and likewise for
# `prod`, `minimum`, `maximum` and `extrema`. `dims` excludes `Colon`, so compiled whole-array
# calls stay valid.
const _ReduceDims = Union{Integer,Tuple{Vararg{Integer}},AbstractVector{<:Integer}}
for fname in (:sum, :prod, :minimum, :maximum, :extrema)
    @eval Base.$(Symbol(:_, fname))(f::F, a::AbstractDiskArray, dims::_ReduceDims; kw...) where {F} =
        $(Symbol(:diskarrays_, fname, :_impl))(f, a, get_backend(a); dims, kw...)
end

# Chunk-wise defaults for the whole-array reductions without `init`. For `any`/`all` they stop
# at the first chunk that decides the result.
for fname in (:sum, :prod, :all, :any, :minimum, :maximum)
    fnameimpl = Symbol("diskarrays_$(fname)_impl")
    fnamedef = Symbol("_diskarrays_$(fname)_default")
    @eval begin
        $(fnameimpl)(f, a::AbstractDiskArray, ::ComputeBackend; dims=:, kw...) =
            $(fnamedef)(f, a; dims, kw...)

        function $(fnamedef)(f, a::AbstractDiskArray; dims=:, kw...)
            if dims === Colon() && isempty(kw)
                $fname(eachchunk(a)) do chunk
                    $fname(f, a[chunk...])
                end
            else
                _diskarrays_reduce_generic($fname, f, a, dims; kw...)
            end
        end
    end
end
diskarrays_extrema_impl(f, a::AbstractDiskArray, ::ComputeBackend; dims=:, kw...) =
    _diskarrays_reduce_generic(extrema, f, a, dims; kw...)

# Base's generic `_sum(f, A, dims; kw...)` and its siblings call `mapreduce`, which reads the
# array chunk by chunk through `diskarrays_mapreduce_impl`. The public `sum(f, A; dims)` would
# come back to the hooks above.
for fname in (:sum, :prod, :minimum, :maximum, :extrema, :any, :all)
    _fname = Symbol(:_, fname)
    @eval _diskarrays_reduce_generic(::typeof($fname), f, a, dims; kw...) =
        invoke(Base.$_fname, Tuple{Any,Any,Any}, f, a, dims; kw...)
end

# `any`/`all` enter through the Base method that invalidates least on each Julia version:
# - 1.11+: Base's internal `_any`/`_all(f, itr, dims)`. A method on `any(f, ::AbstractDiskArray)`
#   invalidates every compiled `any(f, ::Any)` call (553 MethodInstances on 1.12). `f` stays
#   untyped, so callable structs also read chunk by chunk.
# - 1.10: public `any`/`all(f::Function, a)`. There the `_any(f, a, ::Colon)` hook invalidates
#   LinearAlgebra's `_any(f∘transpose, …)` callers (44–66 MethodInstances). `dims` excludes
#   `Colon`, which keeps the `dims` hook unambiguous with Base's `_any(f, itr, ::Colon)`.
@static if VERSION < v"1.11-"
    Base.any(f::F, a::AbstractDiskArray) where {F<:Function} = diskarrays_any_impl(f, a, get_backend(a))
    Base.all(f::F, a::AbstractDiskArray) where {F<:Function} = diskarrays_all_impl(f, a, get_backend(a))
    Base.any(a::AbstractDiskArray) = any(identity, a)
    Base.all(a::AbstractDiskArray) = all(identity, a)
    const _AnyAllDims = _ReduceDims
else
    Base._any(f, a::AbstractDiskArray, ::Colon) = diskarrays_any_impl(f, a, get_backend(a))
    Base._all(f, a::AbstractDiskArray, ::Colon) = diskarrays_all_impl(f, a, get_backend(a))
    const _AnyAllDims = Any
end
# `any(f, a; dims)` and `any(a; dims)`
Base._any(f, a::AbstractDiskArray, dims::_AnyAllDims) = diskarrays_any_impl(f, a, get_backend(a); dims)
Base._all(f, a::AbstractDiskArray, dims::_AnyAllDims) = diskarrays_all_impl(f, a, get_backend(a); dims)

# `count(f, a; dims, init)` and `count(a; dims, init)` both call Base's internal
# `_count(f, a, dims, init)`. The whole-array count with the default `init` calls the hook
# without keywords.
Base._count(f::F, v::AbstractDiskArray, ::Colon, init) where {F} =
    init === 0 ? diskarrays_count_impl(f, v, get_backend(v)) :
    diskarrays_count_impl(f, v, get_backend(v); init)
Base._count(f::F, v::AbstractDiskArray, dims, init) where {F} =
    diskarrays_count_impl(f, v, get_backend(v); dims, init)
function diskarrays_count_impl(f, v::AbstractDiskArray, ::ComputeBackend; dims=:, init=0)
    dims === Colon() || return invoke(Base._count, Tuple{Any,AbstractArray,Any,Any}, f, v, dims, init)
    return foldl(eachchunk(v); init) do n, chunk
        count(f, v[chunk...]; init=n)
    end
end

Base.unique(v::AbstractDiskArray) = unique(identity, v)
Base.unique(f, v::AbstractDiskArray) = diskarrays_unique_impl(f, v, get_backend(v))
function diskarrays_unique_impl(f, v::AbstractDiskArray, ::ComputeBackend)
    reduce((unique(f, v[c...]) for c in eachchunk(v))) do acc, u
        unique!(f, append!(acc, u))
    end
end



# Stubs for functions that will be created once Statistics.jl is loaded
function diskarrays_mean_impl end
function diskarrays_median_impl end