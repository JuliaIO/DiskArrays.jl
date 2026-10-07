
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



# ── Public convenience wrappers (sum, prod, minimum, maximum) ──────────
# These would usually fall back to mapreduce; we define them so that other backends can
# overload `diskarrays_<f>_impl`.
for fname in (:sum, :prod, :minimum, :maximum)
    @eval begin
        # `F` forces specialization on `f`, which is only passed through here
        function Base.$fname(f::F, a::AbstractDiskArray; kwargs...) where {F<:Function}
            $(Symbol("diskarrays_$(fname)_impl"))(f, a, get_backend(a); kwargs...)
        end
        Base.$fname(a::AbstractDiskArray; kwargs...) = Base.$fname(identity, a; kwargs...)
    end
end

# Backend hooks with chunk-wise defaults. For `any`/`all` the default stops at the first
# chunk that decides the result.
for fname in (:sum, :prod, :all, :any, :minimum, :maximum)
    fnameimpl = Symbol("diskarrays_$(fname)_impl")
    fnamedef = Symbol("_diskarrays_$(fname)_default")
    @eval begin
        $(fnameimpl)(f, a::AbstractDiskArray, ::ComputeBackend; dims=:) =
            $(fnamedef)(f, a; dims)

        function $(fnamedef)(f, a::AbstractDiskArray; dims=:)
            if dims === Colon()
                $fname(eachchunk(a)) do chunk
                    $fname(f, a[chunk...])
                end
            else
                _diskarrays_reduce_dims($fname, f, a, dims)
            end
        end
    end
end
# With `dims`, call Base's generic method, which runs into the `mapreducedim!` implementation
# above. For `any`/`all` that is `_any`/`_all(f, A, dims)`: `any(f, A; dims)` itself would
# come back to the `_any` hook below.
_diskarrays_reduce_dims(fname, f, a, dims) =
    invoke(fname, Tuple{typeof(f),AbstractArray{eltype(a),ndims(a)}}, f, a; dims)
_diskarrays_reduce_dims(::typeof(any), f, a, dims) = invoke(Base._any, Tuple{Any,Any,Any}, f, a, dims)
_diskarrays_reduce_dims(::typeof(all), f, a, dims) = invoke(Base._all, Tuple{Any,Any,Any}, f, a, dims)

# `any`/`all` enter through the Base method that invalidates least on each Julia version:
# - 1.11+: Base's internal `_any`/`_all(f, itr, dims)`. A method on `any(f, ::AbstractDiskArray)`
#   invalidates every compiled `any(f, ::Any)` call (553 MethodInstances on 1.12). `f` stays
#   untyped, so callable structs also read chunk by chunk.
# - 1.10: public `any`/`all(f::Function, a)`, which cost 4 there; the `_any(f, a, ::Colon)` hook
#   invalidates LinearAlgebra's `_any(f∘transpose, ::Any, :)` (~70). `dims` excludes `Colon`,
#   which keeps the `dims` hook unambiguous with Base's `_any(f, itr, ::Colon)`.
@static if VERSION < v"1.11-"
    Base.any(f::F, a::AbstractDiskArray) where {F<:Function} = diskarrays_any_impl(f, a, get_backend(a))
    Base.all(f::F, a::AbstractDiskArray) where {F<:Function} = diskarrays_all_impl(f, a, get_backend(a))
    Base.any(a::AbstractDiskArray) = any(identity, a)
    Base.all(a::AbstractDiskArray) = all(identity, a)
    const _AnyAllDims = Union{Integer,Tuple{Vararg{Integer}},AbstractVector{<:Integer}}
else
    Base._any(f, a::AbstractDiskArray, ::Colon) = diskarrays_any_impl(f, a, get_backend(a))
    Base._all(f, a::AbstractDiskArray, ::Colon) = diskarrays_all_impl(f, a, get_backend(a))
    const _AnyAllDims = Any
end
# `any(f, a; dims)` and `any(a; dims)`
Base._any(f, a::AbstractDiskArray, dims::_AnyAllDims) = diskarrays_any_impl(f, a, get_backend(a); dims)
Base._all(f, a::AbstractDiskArray, dims::_AnyAllDims) = diskarrays_all_impl(f, a, get_backend(a); dims)

Base.count(v::AbstractDiskArray) = count(identity, v::AbstractDiskArray)
Base.count(f, v::AbstractDiskArray) = diskarrays_count_impl(f, v, get_backend(v))
function diskarrays_count_impl(f, v::AbstractDiskArray, ::ComputeBackend)
    sum(eachchunk(v)) do chunk
        count(f, v[chunk...])
    end
end

Base.unique(v::AbstractDiskArray) = unique(identity, v)
Base.unique(f, v::AbstractDiskArray) = diskarrays_unique_impl(f, v, get_backend(v))
function diskarrays_unique_impl(f, v::AbstractDiskArray, ::ComputeBackend)
    reduce((unique(f, v[c...]) for c in eachchunk(v))) do acc, u
        unique!(f, append!(acc, u))
    end
end


function Base.extrema(f::F, a::AbstractDiskArray; kwargs...) where {F<:Function}
    diskarrays_extrema_impl(f, a, get_backend(a); kwargs...)
end
Base.extrema(a::AbstractDiskArray; kwargs...) = extrema(identity, a; kwargs...)

diskarrays_extrema_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...) =
    invoke(extrema, Tuple{typeof(f),AbstractArray{eltype(a),ndims(a)}}, f, a; kwargs...)


# Stubs for functions that will be created once Statistics.jl is loaded
function diskarrays_mean_impl end
function diskarrays_median_impl end