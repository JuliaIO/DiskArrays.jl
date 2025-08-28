"""
    unique_threaded(v::AbstractDiskArray)
    unique_threaded(f, v::AbstractDiskArray)

Threaded version of `unique` for DiskArrays.
Only uses threading if the backend is thread-safe and threading is globally enabled.
Falls back to single-threaded implementation otherwise.
"""
function unique_threaded(v::AbstractDiskArray)
    return unique_threaded(identity, v)
end

function unique_threaded(f, v::AbstractDiskArray)
    if !should_use_threading(v)
        # Fall back to single-threaded implementation
        return _unique_single_threaded(f, v)
    end

    chunks = collect(eachchunk(v))
    u = Vector{Vector{eltype(v)}}(undef, length(chunks))

    Threads.@threads :greedy for i in eachindex(chunks)
        chunk = chunks[i]
        u[i] = unique(f, v[chunk...])
    end

    # Reduce results
    return reduce(u; init=eltype(v)[]) do acc, chunk_result
        unique!(f, append!(acc, chunk_result))
    end
end

function _unique_single_threaded(f, v::AbstractDiskArray)
    result = eltype(v)[]
    for chunk in eachchunk(v)
        chunk_unique = unique(f, v[chunk...])
        append!(result, chunk_unique)
        unique!(f, result)
    end
    return result
end

# Extend Base.unique to use threaded version when appropriate
function Base.unique(v::AbstractDiskArray)
    return unique_threaded(v)
end

function Base.unique(f, v::AbstractDiskArray)
    return unique_threaded(f, v)
end
