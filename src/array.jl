# Use broadcast to copy to a new Array
_disk_collect(a::AbstractArray{T,N}) where {T,N} = a[ntuple(_ -> :, Val{N}())...]
_disk_collect(a::AbstractArray{T,0}) where {T} = fill(a[])

# Use broadcast to copy
function _disk_copyto!(dest::AbstractArray{<:Any,N}, source::AbstractArray{<:Any,N}) where {N}
    return dest .= source
end
function _disk_copyto!(dest::AbstractArray, source::AbstractArray)
    # TODO make this more specific so we are reshaping the Non-DiskArray more often.
    reshape(dest, size(source)) .= source
    return dest
end
function _disk_copyto!(dest, Rdest, src, Rsrc)
    if size(Rdest) != size(Rsrc)
        throw(ArgumentError("source and destination must have same size (got $(size(Rsrc)) and $(size(Rdest)))"))
    end

    if isempty(Rdest)
        # This check is here to catch #168
        return dest
    end
    view(dest, Rdest) .= view(src, Rsrc)
    return dest
end
function _disk_copyto_5arg!(dest, dstart, src, sstart, n)
    if iszero(n)
        return dest
    end
    if n < 0
        throw(ArgumentError(LazyString("tried to copy n=",
        n," elements, but n should be non-negative")))
    end
    destv = view(dest, range(dstart, length=n))
    DiskArrays.readblock!(src, destv, range(sstart, length=n))
    return dest
end

# Use a view for lazy reverse
_disk_reverse(a, ::Colon) = _disk_reverse(a, ntuple(identity, ndims(a)))
_disk_reverse(a, dims::Int) = _disk_reverse(a, (dims,))
function _disk_reverse(A, dims::Tuple)
    rev_axes = map(ntuple(identity, ndims(A)), axes(A)) do d, a
        ax = StepRange(a)
        d in dims ? reverse(ax) : ax
    end
    return view(A, rev_axes...)
end

_disk_reverse1(a) = _disk_reverse(a, 1)
function _disk_reverse1(a, start::Int, stop::Int)
    inds = [firstindex(a):start-1; stop:-1:start; stop+1:lastindex(a)]
    return view(a, inds)
end

# Use broadcast instead of a loop. 
# The `count` argument is disallowed as broadcast is not sequential.
function _disk_replace!(new, res::AbstractArray, A::AbstractArray, count::Int)
    count < length(res) &&
        throw(ArgumentError("`replace` on DiskArrays objects cannot use a count value"))
    return broadcast!(new, res, A)
end

macro implement_array_methods(t)
    t = esc(t)
    quote
        Base.Array(a::$t) = $_disk_collect(a)
        Base.collect(a::$t) = $_disk_collect(a)
        Base.copyto!(dest::$t, source::AbstractArray) = $_disk_copyto!(dest, source)
        Base.copyto!(dest::AbstractArray, source::$t) = $_disk_copyto!(dest, source)
        Base.copyto!(dest::$t, source::$t) = $_disk_copyto!(dest, source)
        function Base.copyto!(
            dest::$t, Rdest::CartesianIndices, src::AbstractArray, Rsrc::CartesianIndices
        )
            return $_disk_copyto!(dest, Rdest, src, Rsrc)
        end
        function Base.copyto!(
            dest::AbstractArray, Rdest::CartesianIndices, src::$t, Rsrc::CartesianIndices
        )
            return $_disk_copyto!(dest, Rdest, src, Rsrc)
        end
        function Base.copyto!(
            dest::$t, Rdest::CartesianIndices, src::$t, Rsrc::CartesianIndices
        )
            return $_disk_copyto!(dest, Rdest, src, Rsrc)
        end
        # For ambiguity
        Base.copyto!(dest::PermutedDimsArray, src::$t) = DiskArrays._copyto!(dest, src)
        function Base.copyto!(dest::PermutedDimsArray{T,N}, src::$t{T,N}) where {T,N}
            return $_disk_copyto!(dest, src)
        end
        function Base.copyto!(dest::Vector, dstart::Integer, src::$t{<:Any, 1}, sstart::Integer, n::Integer)
            return $_disk_copyto_5arg!(dest, dstart, src, sstart, n)
        end
        function Base.copyto!(dest::SubArray{T, 1, Vector{T}, <:Tuple{AbstractUnitRange}, true} where {T}, dstart::Integer, src::$t{<:Any, 1}, sstart::Integer, n::Integer)
            return $_disk_copyto_5arg!(dest, dstart, src, sstart, n)
        end

        Base.reverse(a::$t; dims=:) = $_disk_reverse(a, dims)
        Base.reverse(a::$t{<:Any,1}) = $_disk_reverse1(a)
        Base.reverse(a::$t{<:Any,1}, start::Integer, stop::Integer=lastindex(a)) =
            $_disk_reverse1(a, start, stop)

        # Here we extend the unexported `_replace` method, but we replicate 
        # much less Base functionality by extending it rather than `replace`.
        function Base._replace!(new::Base.Callable, res::AbstractArray, A::$t, count::Int)
            return $_disk_replace!(new, res, A, count)
        end
        function Base._replace!(new::Base.Callable, res::$t, A::AbstractArray, count::Int)
            return $_disk_replace!(new, res, A, count)
        end
        function Base._replace!(new::Base.Callable, res::$t, A::$t, count::Int)
            return $_disk_replace!(new, res, A, count)
        end
    end
end

