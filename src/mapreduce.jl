
# Implementation macro

macro implement_mapreduce(t)
    t = esc(t)
    quote
        function Base._mapreduce(f, op, ::IndexCartesian, v::$t)
            mapreduce(op, eachchunk(v)) do cI
                a = v[to_ranges(cI)...]
                mapreduce(f, op, a)
            end
        end
        function Base.mapreducedim!(f, op, R::AbstractArray, A::$t)
            sR = size(R)
            foreach(eachchunk(A)) do cI
                a = A[to_ranges(cI)...]
                ainds = map(
                    (cinds, arsize) -> arsize == 1 ? Base.OneTo(1) : cinds,
                    to_ranges(cI),
                    size(R),
                )
                # Maybe the view into R here is problematic and a copy would be faster
                Base.mapreducedim!(f, op, view(R, ainds...), a)
            end
            return R
        end

        function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::$t)
            cc = eachchunk(itr)
            isempty(cc) &&
                return Base.mapreduce_empty_iter(f, op, itr, Base.IteratorEltype(itr))
            return Base.mapfoldl_impl(f, op, nt, itr, cc)
        end
        function Base.mapfoldl_impl(f, op, nt::NamedTuple{()}, itr::$t, cc)
            y = first(cc)
            a = itr[to_ranges(y)...]
            init = mapfoldl(f, op, a)
            return Base.mapfoldl_impl(f, op, (init=init,), itr, Iterators.drop(cc, 1))
        end
        function Base.mapfoldl_impl(f, op, nt::NamedTuple{(:init,)}, itr::$t, cc)
            init = nt.init
            for y in cc
                a = itr[to_ranges(y)...]
                init = mapfoldl(f, op, a; init=init)
            end
            return init
        end
    end
end


# Implementation for special cases and if fallback breaks in future julia versions

for fname in [:sum, :prod, :all, :any, :minimum, :maximum]
    @eval Base.$fname(v::AbstractDiskArray) = Base.$fname(identity, v::AbstractDiskArray)
    @eval function Base.$fname(f::Function, v::AbstractDiskArray)
        $fname(eachchunk(v)) do chunk
            $fname(f, v[chunk...])
        end
    end
end

Base.count(v::AbstractDiskArray) = count(identity, v::AbstractDiskArray)
function Base.count(f, v::AbstractDiskArray)
    sum(eachchunk(v)) do chunk
        count(f, v[chunk...])
    end
end

Base.unique(v::AbstractDiskArray) = unique(identity, v)
function Base.unique(f, v::AbstractDiskArray)
    reduce((unique(f, v[c...]) for c in eachchunk(v))) do acc, u
        unique!(f, append!(acc, u))
    end
end
