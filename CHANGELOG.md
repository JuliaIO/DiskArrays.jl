# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

### Changed

- Iteration yields values in the column-major order of Base arrays, so `zip`, generators, `enumerate`,
  `Iterators.take`/`drop`, `foldl` and `accumulate` pair values with the right indices. Values are read in blocks of
  whole chunks that fit in `default_chunk_size`; each chunk is read once when a block can span the last dimension.
- `zip` with a disk array accepts any iterator.

### Deprecated

- `@implement_zip` and `@implement_generator` do nothing, and `@implement_diskarray_skip_zip` is
  `@implement_diskarray`. Calling them emits a deprecation warning.

### Removed

- The internal `DiskZip`, `DiskGenerator` and `BlockedIndices` types, and the `Base.zip`,
  `Base.Generator` and `Base.eachindex` methods on disk arrays, which invalidated much compiled code
  ([#175](https://github.com/JuliaIO/DiskArrays.jl/issues/175)).

- Initial release

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/JuliaIO/DiskArrays.jl/compare/v0.1.0...HEAD
