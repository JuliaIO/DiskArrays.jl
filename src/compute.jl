using Preferences

abstract type ComputeBackend end

#Fallback: get_backend returns the backend itself
get_backend(b::ComputeBackend) = b

# Default Backend, simple nonthreaded, non-distributed computations
struct DefaultBackend <: ComputeBackend end

# Backend type for DiskArrayEngine.jl, which implements the `diskarrays_*_impl` methods for it
struct DiskArrayEngineBackend <: ComputeBackend end

# Wrapper type that lets users switch backends dynamically as the program runs.
# The field is a small Union of concrete types, so dispatch on `get_backend` is
# union-split into a branch instead of a dynamic dispatch. Backends that are not
# known to DiskArrays can be attached to an array via `withbackend`.
mutable struct DynamicBackend <: ComputeBackend
    current_backend::Union{DefaultBackend,DiskArrayEngineBackend}
end
get_backend(b::DynamicBackend) = b.current_backend

const backend = @load_preference("backend", "default")

function set_backend(new_backend::String)
    if !(new_backend in ("default", "dynamic", "DiskArrayEngine"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

function load_backend()
    @static if backend == "default"
        DefaultBackend()
    elseif backend == "dynamic"
        return DynamicBackend(DefaultBackend())
    elseif backend == "DiskArrayEngine"
        return DiskArrayEngineBackend()
    else
        return nothing
    end
end
const compute_backend = load_backend()

"""
    set_dynamic_backend!(backend::Union{DefaultBackend,DiskArrayEngineBackend})

Switch the active backend at runtime. Requires the `"dynamic"` backend preference,
see [`set_backend`](@ref).
"""
function set_dynamic_backend!(b::Union{DefaultBackend,DiskArrayEngineBackend})
    compute_backend isa DynamicBackend ||
        error("Runtime switching requires `DiskArrays.set_backend(\"dynamic\")`, current backend is \"$backend\"")
    compute_backend.current_backend = b
end

function _backend_error_hint(io, exc, argtypes, kwargs)
    any(T -> T <: DiskArrayEngineBackend, argtypes) || return
    print(io, "\n`DiskArrayEngineBackend` methods are implemented in DiskArrayEngine.jl, make sure it is loaded with `using DiskArrayEngine`.")
end
