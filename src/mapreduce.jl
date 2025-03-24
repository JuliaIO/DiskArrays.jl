
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

for (fname, _fname) in ((:sum, :_sum), (:prod, :_prod),
                        (:all, :_all), (:any, :_any),
                        (:minimum, :_minimum), (:maximum, :_maximum))
    @eval Base.$fname(v::AbstractDiskArray; dims=:) = Base.$_fname(v, dims)
    @eval Base.$fname(f::Function, v::AbstractDiskArray; dims=:) = Base.$_fname(f, v, dims)
    @eval Base.$_fname(v::AbstractDiskArray, ::Colon) = Base.$_fname(identity, v, :)
    @eval function Base.$_fname(f::Function, v::AbstractDiskArray, ::Colon)
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

for (_fname, init, acum) in ((:_sum, :zero, :+), (:_prod, :one, :*),
                             (:_all, _->:true, :&), (:_any, _->:false, :|),
                             (:_maximum, :typemin, :max), (:_minimum, :typemax, :min))
	@eval Base.$_fname(a::AbstractDiskArray, dims)  = Base.$_fname(identity, a, dims)
	@eval function Base.$_fname(f::Function, a::AbstractDiskArray, dims)
		_dims = typeof(dims)<:Tuple ? [dims...] : typeof(dims)<:Number ? [dims] : dims
		out_dims = [size(a)...]
		out_dims[_dims] .= 1
		out = fill($init(Base.return_types(f, (eltype(a),))[1]), out_dims...)
		for c in eachchunk(a)
			out_c = [c...]
			out_c[_dims] .= Ref(1:1)
			out[out_c...] .= $acum.(out[out_c...], Base.$_fname(f, a[c...], dims))
		end
		return out
	end
end
