module StatisticsExt
import Statistics
import DiskArrays: AbstractDiskArray, ComputeBackend, get_backend, diskarrays_mean_impl

function Statistics.mean(f::F, a::AbstractDiskArray; kwargs...) where {F<:Function}
    diskarrays_mean_impl(f, a, get_backend(a); kwargs...)
end
Statistics.mean(a::AbstractDiskArray; kwargs...) = Statistics.mean(identity, a; kwargs...)

diskarrays_mean_impl(f, a::AbstractDiskArray, ::ComputeBackend;kwargs...) =
    invoke(Statistics.mean,Tuple{typeof(f),AbstractArray{eltype(a),ndims(a)}},f,a;kwargs...)





end