"""
    AllowStepRange

Traits to specify if an array axis can utilise step ranges,
as an argument to `BatchStrategy` types `NoBatch`, `SubRanges`
and `ChunkRead`.

`CanStepRange()` and `NoStepRange()` are the two options.
"""
abstract type AllowStepRange end

struct CanStepRange <: AllowStepRange end
struct NoStepRange <: AllowStepRange end

"""
    BatchStrategy{S<:AllowStepRange}
    
Traits for array chunking strategy.

`NoBatch`, `SubRanges` and `ChunkRead` are the options.

All have keywords:

- `alow_steprange`: an [`AllowStepRange`](@ref) trait, NoStepRange() by default.
    this controls if step range are passed to the parent object.
- `density_threshold`: determines the density where step ranges are not read as whole chunks.
"""
abstract type BatchStrategy{S<:AllowStepRange} end

"""
    NoBatch <: BatchStrategy

A chunking strategy that avoids batching into multiple reads.
"""
@kwdef struct NoBatch{S} <: BatchStrategy{S}
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
NoBatch(from::BatchStrategy) =
    NoBatch(from.allow_steprange, from.density_threshold)

"""
    SubRanges <: BatchStrategy

A chunking strategy that splits contiguous streaks 
into ranges to be read separately.

A vector of indices is sorted, split into runs of consecutive indices (or,
with `CanStepRange()`, runs with a constant step) that are each read as one
range, and reassembled in the requested order, so the vector need not be
sorted or unique. For example `[12, 5]` reads the single range `5:7:12` with
`CanStepRange()` and the ranges `5:5` and `12:12` with `NoStepRange()`.
"""
@kwdef struct SubRanges{S} <: BatchStrategy{S}
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
SubRanges(from::BatchStrategy) =
    SubRanges(from.allow_steprange, from.density_threshold)
"""
    ChunkRead <: BatchStrategy

A chunking strategy splits a dataset according to chunk,
and reads chunk by chunk.
"""
@kwdef struct ChunkRead{S} <: BatchStrategy{S}
    allow_steprange::S = NoStepRange()
    density_threshold::Float64 = 0.5
end
ChunkRead(from::BatchStrategy) =
    ChunkRead(from.allow_steprange, from.density_threshold)

