
"""
    ConcatDiskArray <: AbstractDiskArray
    
    ConcatDiskArray(arrays)

Joins multiple `AbstractArray`s or `AbstractDiskArray`s into
a single disk array, using lazy concatination.

Returned from `cat` on disk arrays. 

It is also useful on its own as it can easily concatenate an array of disk arrays.
"""
struct ConcatDiskArray{T,N,P,C,HC} <: AbstractDiskArray{T,N}
    parents::P
    startinds::NTuple{N,Vector{Int}}
    size::NTuple{N,Int}
    chunks::C
    haschunks::HC
end
function ConcatDiskArray(arrays::AbstractArray{<:AbstractArray{<:Any,N},M}) where {N,M}
    T = mapreduce(eltype, promote_type, init=eltype(first(arrays)), arrays)

    if N > M
        newshape = extenddims(size(arrays), size(first(arrays)), 1)
        arrays1 = reshape(arrays, newshape)
        D = N
    else
        arrays1 = arrays
        D = M
    end
    startinds, sizes = arraysize_and_startinds(arrays1)

    chunks = concat_chunksize(arrays1)
    hc = Chunked(batchstrategy(chunks))

    return ConcatDiskArray{T,D,typeof(arrays1),typeof(chunks),typeof(hc)}(arrays1, startinds, sizes, chunks, hc)
end
function ConcatDiskArray(arrays::AbstractArray)
    # Validate array eltype and dimensionality
    all(a -> ndims(a) == ndims(first(arrays)), arrays) ||
        error("Arrays don't have the same dimensions")
    return error("Should not be reached")
end
extenddims(a::Tuple{Vararg{Any,N}}, b::Tuple{Vararg{Any,M}}, fillval) where {N,M} = extenddims((a..., fillval), b, fillval)
extenddims(a::Tuple{Vararg{Any,N}}, _::Tuple{Vararg{Any,N}}, _) where {N} = a

Base.size(a::ConcatDiskArray) = a.size

function arraysize_and_startinds(arrays1)
    sizes = map(i -> zeros(Int, i), size(arrays1))
    for i in CartesianIndices(arrays1)
        ai = arrays1[i]
        sizecur = extenddims(size(ai), size(arrays1), 1)
        foreach(sizecur, i.I, sizes) do si, ind, sizeall
            if sizeall[ind] == 0
                #init the size
                sizeall[ind] = si
            elseif sizeall[ind] != si
                throw(ArgumentError("Array sizes don't form a grid"))
            end
        end
    end
    r = map(sizes) do sizeall
        pushfirst!(sizeall, 1)
        for i in 2:length(sizeall)
            sizeall[i] = sizeall[i-1] + sizeall[i]
        end
        pop!(sizeall) - 1, sizeall
    end
    map(last, r), map(first, r)
end

# DiskArrays interface

eachchunk(a::ConcatDiskArray) = a.chunks
haschunks(c::ConcatDiskArray) = c.haschunks

function readblock!(a::ConcatDiskArray, aout, inds::AbstractUnitRange...)
    # Find affected blocks and indices in blocks
    _concat_diskarray_block_io(a, inds...) do outer_range, array_range, I
        vout = view(aout, outer_range...)
        readblock!(a.parents[I], vout, array_range...)
    end
end
function writeblock!(a::ConcatDiskArray, aout, inds::AbstractUnitRange...)
    _concat_diskarray_block_io(a, inds...) do outer_range, array_range, I
        data = view(aout, outer_range...)
        writeblock!(a.parents[I], data, array_range...)
    end
end

# Utils

function _concat_diskarray_block_io(f, a::ConcatDiskArray, inds...)
    # Find affected blocks and indices in blocks
    blockinds = map(inds, a.startinds, size(a.parents)) do i, si, s
        bi1 = max(searchsortedlast(si, first(i)), 1)
        bi2 = min(searchsortedfirst(si, last(i) + 1) - 1, s)
        bi1:bi2
    end
    map(CartesianIndices(blockinds)) do cI
        myar = a.parents[cI]
        mysize = extenddims(size(myar), cI.I, 1)
        array_range = map(cI.I, a.startinds, mysize, inds) do ii, si, ms, indstoread
            max(first(indstoread) - si[ii] + 1, 1):min(last(indstoread) - si[ii] + 1, ms)
        end
        outer_range = map(cI.I, a.startinds, array_range, inds) do ii, si, ar, indstoread
            (first(ar)+si[ii]-first(indstoread)):(last(ar)+si[ii]-first(indstoread))
        end
        #Shorten array range to shape of actual array
        array_range = map((i, j) -> j, size(myar), array_range)
        outer_range = fix_outerrangeshape(outer_range, array_range)
        f(outer_range, array_range, cI)
    end
end
fix_outerrangeshape(outer_range, array_range) = fix_outerrangeshape((), outer_range, array_range)
fix_outerrangeshape(res, outer_range, array_range) =
    fix_outerrangeshape((res..., first(outer_range)), Base.tail(outer_range), Base.tail(array_range))
fix_outerrangeshape(res, outer_range, ::Tuple{}) =
    fix_outerrangeshape((res..., only(first(outer_range))), Base.tail(outer_range), ())
fix_outerrangeshape(res, ::Tuple{}, ::Tuple{}) = res


function concat_chunksize(parents)
    newchunks = map(s -> Vector{Union{RegularChunks,IrregularChunks}}(undef, s), size(parents))
    for i in CartesianIndices(parents)
        array = parents[i]
        chunks = eachchunk(array)
        foreach(chunks.chunks, i.I, newchunks) do c, ind, newc
            if !isassigned(newc, ind)
                newc[ind] = c
            elseif c != newc[ind]
                throw(ArgumentError("Chunk sizes don't forma grid"))
            end
        end
    end
    newchunks = map(newchunks) do v
        #Chunks that have not been set are from additional dimensions in the parent array shape
        for i in eachindex(v)
            if !isassigned(v, i)
                v[i] = RegularChunks(1, 0, 1)
            end
        end
        # Merge the chunks
        init = RegularChunks(approx_chunksize(first(v)), 0, 0)
        reduce(mergechunks, v; init=init)
    end
    extenddims(newchunks, size(parents), RegularChunks(1, 0, 1))
    return GridChunks(newchunks...)
end

function mergechunks(a::RegularChunks, b::RegularChunks)
    if a.arraysize == 0 || (a.chunksize == b.chunksize && length(last(a)) == a.chunksize)
        RegularChunks(a.chunksize, a.offset, a.arraysize + b.arraysize)
    else
        mergechunks_irregular(a, b)
    end
end
mergechunks(a::ChunkVector, b::ChunkVector) = mergechunks_irregular(a, b)

mergechunks_irregular(a, b) =
    IrregularChunks(; chunksizes=filter(!iszero, [length.(a); length.(b)]))

function cat_disk(dims, As::AbstractArray...)
    if length(dims) == 1
        dims = only(dims)
        cat_disk(dims, As...)
    else
        throw(ArgumentError("Block concatenation is not yet implemented for DiskArrays."))
    end
end
function cat_disk(dims::Int, As::AbstractArray...)
    sz = map(ntuple(identity, dims)) do i
        i == dims ? length(As) : 1
    end
    cdas = reshape(collect(As), sz)
    return ConcatDiskArray(cdas)
end

# Implementation macro

macro implement_cat(t)
    t = esc(t)
    quote
        # Allow mixed lazy cat of other arrays and disk arrays to still be lazy
        # TODO this could be better. allowing non-AbstractDiskArray in
        # the macro makes this kind of impossible to avoid dispatch problems
        Base.cat(A1::$t, As::AbstractArray...; dims) = cat_disk(dims, A1, As...)
        function Base.cat(A1::AbstractArray, A2::$t, As::AbstractArray...; dims)
            return cat_disk(dims, A1, A2, As...)
        end
        function Base.cat(A1::$t, A2::$t, As::AbstractArray...; dims)
            return cat_disk(dims, A1, A2, As...)
        end
        function Base.vcat(
            A1::Union{$t{<:Any,1},$t{<:Any,2}}, As::Union{$t{<:Any,1},$t{<:Any,2}}...
        )
            return cat_disk(1, A1, As...)
        end
        function Base.hcat(
            A1::Union{$t{<:Any,1},$t{<:Any,2}}, As::Union{$t{<:Any,1},$t{<:Any,2}}...
        )
            return cat_disk(2, A1, As...)
        end
    end
end
