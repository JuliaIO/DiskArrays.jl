"""
    withbackend(a::AbstractArray, backend::ComputeBackend)

Wrap `a` so that computations on it use `backend`, regardless of the global backend.
"""
withbackend(a::AbstractArray, backend::ComputeBackend) = WithBackendDiskArray(a, backend)

struct WithBackendDiskArray{T,N,A<:AbstractArray{T,N},B<:ComputeBackend} <: AbstractDiskArray{T,N}
    parent::A
    backend::B
end
Base.parent(a::WithBackendDiskArray) = a.parent
Base.size(a::WithBackendDiskArray) = size(parent(a))
haschunks(a::WithBackendDiskArray) = haschunks(parent(a))
eachchunk(a::WithBackendDiskArray) = eachchunk(parent(a))
readblock!(a::WithBackendDiskArray, aout, i::OrdinalRange...) = readblock!(parent(a), aout, i...)
writeblock!(a::WithBackendDiskArray, v, i::OrdinalRange...) = writeblock!(parent(a), v, i...)

# Find the backend of an array: walk up the parents looking for a `withbackend`
# wrapper and fall back to the global backend. `parent(a) === a` marks the root.
get_backend(a::WithBackendDiskArray) = a.backend
function get_backend(a::AbstractArray)
    p = parent(a)
    return p === a ? get_backend(compute_backend) : get_backend(p)
end
