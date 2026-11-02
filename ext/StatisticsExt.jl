module StatisticsExt
using Statistics: Statistics
import DiskArrays: AbstractDiskArray, ComputeBackend, get_backend, diskarrays_mean_impl

function Statistics.mean(f::F, a::AbstractDiskArray; kwargs...) where {F<:Function}
    return diskarrays_mean_impl(f, a, get_backend(a); kwargs...)
end
Statistics.mean(a::AbstractDiskArray; kwargs...) = Statistics.mean(identity, a; kwargs...)

function diskarrays_mean_impl(f, a::AbstractDiskArray, ::ComputeBackend; kwargs...)
    return invoke(
        Statistics.mean, Tuple{typeof(f),AbstractArray{eltype(a),ndims(a)}}, f, a; kwargs...
    )
end

end
