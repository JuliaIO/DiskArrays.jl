using Preferences

abstract type ComputeBackend end

#Fallback: get_backend returns the backend itself
get_backend(b::ComputeBackend) = b

# Default Backend, simple nonthreaded, non-distributed computations
struct DefaultBackend <: ComputeBackend end

# Backend type for DiskArrayEngine.jl, which implements the `diskarrays_*_impl` methods for it
struct DiskArrayEngineBackend <: ComputeBackend end

# The global backend is always wrapped in a `DynamicBackend` so it can be switched at runtime.
# The field is a small Union of concrete types, so dispatch on `get_backend` is
# union-split into a branch instead of a dynamic dispatch. Backends that are not
# known to DiskArrays can be attached to an array via `withbackend`.
mutable struct DynamicBackend <: ComputeBackend
    current_backend::Union{DefaultBackend,DiskArrayEngineBackend}
end
get_backend(b::DynamicBackend) = b.current_backend

# The preference only selects the backend that is active when DiskArrays is loaded
const backend = @load_preference("backend", "default")

function _backend_from_name(name::String)
    name == "default" && return DefaultBackend()
    name == "DiskArrayEngine" && return DiskArrayEngineBackend()
    throw(ArgumentError("Invalid backend: \"$(name)\""))
end

const compute_backend = DynamicBackend(_backend_from_name(backend))

"""
    set_dynamic_backend!(backend::Union{DefaultBackend,DiskArrayEngineBackend})

Switch the active backend for the current session.
"""
set_dynamic_backend!(b::Union{DefaultBackend,DiskArrayEngineBackend}) =
    compute_backend.current_backend = b

"""
    set_backend(new_backend::String)

Switch the active backend to `"default"` or `"DiskArrayEngine"` and save it as
a preference, so it is also the active backend in future sessions.
"""
function set_backend(new_backend::String)
    b = _backend_from_name(new_backend)
    @set_preferences!("backend" => new_backend)
    set_dynamic_backend!(b)
end

function _backend_error_hint(io, exc, argtypes, kwargs)
    any(T -> T <: DiskArrayEngineBackend, argtypes) || return
    print(io, "\n`DiskArrayEngineBackend` methods are implemented in DiskArrayEngine.jl, make sure it is loaded with `using DiskArrayEngine`.")
end
