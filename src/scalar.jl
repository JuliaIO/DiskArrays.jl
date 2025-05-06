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
checkscalar(::Type{Bool}, A::AbstractArray, ::Tuple{}) = true # Handle 0 dimensional
checkscalar(::Type{Bool}, A::AbstractArray, I::Tuple) = !all(map(i -> i isa Int, I)) || canscalar()
checkscalar(::Type{Bool}, A::AbstractArray, I...) = checkscalar(Bool, A, (I...,))
checkscalar(A::AbstractArray, I::Tuple) = checkscalar(Bool, A, I::Tuple) || _scalar_error()
checkscalar(A::AbstractArray, I...) = checkscalar(A, I)

function _scalar_error()
    return error(
        "Scalar indexing with `Int` is very slow, and currently is disallowed. Run DiskArrays.allowscalar(true) to allow",
    )
end
