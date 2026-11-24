"""
    DiskGenerator

Alias for `Base.Generator` over an `AbstractDiskArray`, e.g. `(f(x) for x in disk_array)`.

Iteration is unchanged, but `collect` (and so array comprehensions) reads the
underlying array chunk by chunk, out of order, and writes the result into
the correct position of the output array.

This is implemented by specializing `collect` on the generator type rather than
by overriding the `Base.Generator` constructor, which would invalidate
compiled code for every `Base.Generator(f, ::Any)` call.
"""
const DiskGenerator = Base.Generator{<:AbstractDiskArray}

# Fill `dest`, allocated by `alloc(eltype, axes)`, with the generator output in the
# order the disk array iterates (chunk by chunk), placing each value at its own index.
# Copied from `collect(::Generator)` in julia 1.9
function _collect_disk_generator(alloc, itr::Base.Generator)
    y = iterate(itr)
    shp = axes(itr.iter)
    if y === nothing
        et = Base.@default_eltype(itr)
        return alloc(et, shp)
    end
    v1, st = y
    dest = alloc(typeof(v1), shp)
    i = y
    for I in eachindex(itr.iter)
        if i isa Nothing # Mainly to keep JET clean
            error(
                "Should not be reached: iterator is shorter than its `eachindex` iterator"
            )
        else
            dest[I] = first(i)
            i = iterate(itr, last(i))
        end
    end
    return dest
end

_collect_disk_generator(itr::Base.Generator) =
    _collect_disk_generator(itr) do et, shp
        similar(Array{et,length(shp)}, shp)
    end

_collect_similar_disk_generator(A::AbstractArray, itr::Base.Generator) =
    _collect_disk_generator((et, shp) -> similar(A, et, shp), itr)

# Note: these extend `collect`/`map` instead of adding methods to the `Base.Generator`
# constructor. The latter invalidates every compiled `Base.Generator(f, ::Any)` call site,
# which is a lot of code in Base and the stdlibs.
macro implement_generator(t)
    t = esc(t)
    quote
        function Base.collect(itr::Base.Generator{<:$t})
            return $_collect_disk_generator(itr)
        end
        # `Base.map(f, A::AbstractArray)` is `collect_similar(A, Generator(f, A))`,
        # which would otherwise fill the output in chunk order, not in index order.
        function Base.map(f, A::$t)
            return $_collect_similar_disk_generator(A, Base.Generator(f, A))
        end
    end
end
