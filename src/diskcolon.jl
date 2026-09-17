"""
    DiskColon()

A custom index type that behaves exactly like `Colon()` when indexing a disk array.

It subtypes `AbstractVector{Int}` only so that wrappers which accept a closed set of index
types (e.g. DimensionalData.jl) pass it through. Like `Colon`, it has no size until
`to_indices` resolves it against an axis, to a [`DiskSlice`](@ref).
"""
struct DiskColon <: AbstractVector{Int} end

"""
    DiskSlice(indices)

A `DiskColon` resolved to the axis it indexes, as `Base.Slice` is for `Colon`.
An `AbstractUnitRange`, so it indexes anything a range can.
"""
struct DiskSlice{R<:AbstractUnitRange{<:Integer}} <: AbstractUnitRange{Int}
    indices::R
end
Base.first(s::DiskSlice) = Int(first(s.indices))
Base.last(s::DiskSlice) = Int(last(s.indices))
Base.show(io::IO, s::DiskSlice) = print(io, "DiskSlice(", s.indices, ")")

Base.show(io::IO, ::DiskColon) = print(io, "DiskColon()")
Base.show(io::IO, ::MIME"text/plain", c::DiskColon) = show(io, c)
Base.checkindex(::Type{Bool}, ::AbstractUnitRange, ::DiskColon) = true
Base.to_indices(A, inds, I::Tuple{DiskColon,Vararg}) =
    (DiskSlice(to_indices(A, inds, (:,))[1]), to_indices(A, Base.safe_tail(inds), tail(I))...)

# `getindex`/`setindex!` on a disk array never call `to_indices`, so an unresolved `DiskColon`
# gets here. `DiskSlice` needs nothing: it takes the `AbstractUnitRange` methods.
_need_batch_index(::DiskColon, chunks, batch_strategy) = _need_batch_index(:, chunks, batch_strategy)
# One method per strategy and array type, to be more specific than the `AbstractArray{<:Integer}` methods
for S in (:NoBatch, :ChunkRead, :SubRanges)
    @eval process_index(::DiskColon, chunks::Tuple{Vararg{ChunkVector}}, batch_strategy::$S) =
        process_index(:, chunks, batch_strategy)
end
for A in (:Any, :AbstractVector)
    @eval DiskIndex(a::$A, ::Tuple{DiskColon}, batch_strategy) = DiskIndex(a, (:,), batch_strategy)
end
