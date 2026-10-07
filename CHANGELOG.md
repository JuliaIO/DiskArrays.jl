# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- `estimate_chunksize`, and so `eachchunk` of an in-memory array or view, no longer allocates
- `element_size` also accepts an element type, `element_size(T::Type)`, so callers can
  size buffers before an array exists; `element_size(a::AbstractArray)` now forwards to it

- Initial release

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/JuliaIO/DiskArrays.jl/compare/v0.1.0...HEAD
