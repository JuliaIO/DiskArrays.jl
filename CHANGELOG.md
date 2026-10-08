# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- Reduce method invalidations when loading DiskArrays (roughly 900 fewer invalidated
  `MethodInstance`s in a fresh session):
  - Generators over disk arrays no longer override the `Base.Generator` constructor.
    `DiskArrays.DiskGenerator` is now an alias for `Base.Generator{<:AbstractDiskArray}`
    and chunk-ordered `collect` is implemented by specializing `collect` and `map`.
    `@implement_generator` now defines these methods for the given type.
  - `zip` of a disk array with a non-array iterator no longer throws an `ArgumentError`,
    and falls back to `Base.zip`.
  - `ChunkIndices` has the correct `eltype` from its supertype, so the `eltype` method is removed.
- The `AbstractDiskArray` interface (`readblock!`, `writeblock!`, `eachchunk`, `haschunks`,
  `chunkexists`), the chunk types (`GridChunks`, `RegularChunks`, `Chunked`, `ChunkIndex`, ...) and
  the `BatchStrategy` types moved to the new dependency-free subpackage `DiskArraysCore`
  (`lib/DiskArraysCore`). DiskArrays re-exports them: `DiskArrays.AbstractDiskArray === DiskArraysCore.AbstractDiskArray`
  and nothing changes for users. Packages that only want to subtype `AbstractDiskArray`
  can depend on DiskArraysCore and so avoid loading DiskArrays' generic array methods.

## v0.4.25

- `estimate_chunksize`, and so `eachchunk` of an in-memory array or view, no longer allocates
- `element_size` also accepts an element type, `element_size(T::Type)`, so callers can
  size buffers before an array exists; `element_size(a::AbstractArray)` now forwards to it

- Initial release

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/JuliaIO/DiskArrays.jl/compare/v0.1.0...HEAD
