# Manual control over scalar indexing
const ALLOWSCALAR = Ref{Bool}(true)

"""
    allowscalar(x::Bool)

Specify if a disk array can do scalar indexing, (with all `Int` arguments).

Setting `allowscalar(false)` can help identify the cause of poor performance.
"""
allowscalar(x::Bool) = ALLOWSCALAR[] = x

"""
    canscalar()

Check if DiskArrays is set to allow scalar indexing, with [`allowscalar`](@ref).

Returns a `Bool`.
"""
canscalar() = ALLOWSCALAR[]

@deprecate allow_scalar allowscalar
@deprecate can_scalar canscalar

# Checks if an index is scalar at all, and then if scalar indexing is allowed. 
# Syntax as for `checkbounds`.
checkscalar(::Type{Bool}, a::AbstractArray, i::Integer...) = checkscalar(Bool, i...)
checkscalar(::Type{Bool}) = true # Handle 0 dimensional
checkscalar(::Type{Bool}, I::Tuple) = checkscalar(Bool, I...)
checkscalar(::Type{Bool}, I::Integer...) = !all(map(i -> i isa Int, I)) || canscalar()
checkscalar(A::AbstractArray, I::Tuple) = checkscalar(A, I...)
checkscalar(A::AbstractArray, i::Integer...) = checkscalar(Bool, A, i...) || _scalar_error()
checkscalar(I::Tuple) = checkscalar(I...)
checkscalar(i::Integer...) = checkscalar(Bool, i...) || _scalar_error()

function _scalar_error()
    return error(
        "Scalar indexing with `Int` is very slow, and currently is disallowed. Run DiskArrays.allowscalar(true) to allow",
    )
end
