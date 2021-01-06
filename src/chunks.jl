"""
    eachchunk(a)

Returns an iterator with `CartesianIndices` elements that mark the index range of each chunk within an array.
"""
function eachchunk end

struct GridChunks{N}
    parentsize::NTuple{N,Int}
    chunksize::NTuple{N,Int}
    chunkgridsize::NTuple{N,Int}
    offset::NTuple{N,Int}
end
GridChunks(a, chunksize; offset = (_->0).(size(a))) = GridChunks(Int.(size(a)), Int.(chunksize), Int.(getgridsize(size(a),chunksize,offset)),Int.(offset))
GridChunks(a::Tuple, chunksize; offset = (_->0).(a)) = GridChunks(Int.(a), Int.(chunksize), Int.(getgridsize(a,chunksize,offset)),Int.(offset))
function getgridsize(a,chunksize,offset)
  map(a,chunksize,offset) do s,cs,of
    fld1(s+of,cs)
  end
end
function Base.show(io::IO, g::GridChunks)
  print(io,"Regular ",join(g.chunksize,"x")," chunks over a ", join(g.parentsize,"x"), " array.")
end
Base.size(g::GridChunks) = g.chunkgridsize
Base.size(g::GridChunks, dim) = g.chunkgridsize[dim]
Base.IteratorSize(::Type{GridChunks{N}}) where N = Base.HasShape{N}()
Base.eltype(::Type{GridChunks{N}}) where N = CartesianIndices{N,NTuple{N,UnitRange{Int64}}}
Base.length(c::GridChunks) = prod(size(c))
@inline function _iterate(g,r)
    if r === nothing
        return nothing
    else
        ichunk, state = r
        outinds = map(ichunk.I, g.chunksize, g.parentsize,g.offset) do ic, cs, ps, of
            max((ic-1)*cs+1-of,1):min(ic*cs-of, ps)
        end |> CartesianIndices
        outinds, state
    end
end
function Base.iterate(g::GridChunks)
    r = iterate(CartesianIndices(g.chunkgridsize))
    _iterate(g,r)
end
function Base.iterate(g::GridChunks, state)
    r = iterate(CartesianIndices(g.chunkgridsize), state)
    _iterate(g,r)
end

#Define the approx default maximum chunk size (in MB)
"The target chunk size for processing for unchunked arrays in MB, defaults to 100MB"
const default_chunk_size = Ref(100)

"""
A fallback element size for arrays to determine a where elements have unknown
size like strings. Defaults to 100MB
"""
const fallback_element_size = Ref(100)

#Here we implement a fallback chunking for a DiskArray although this should normally
#be over-ridden by the package that implements the interface

function eachchunk(a::AbstractArray)
  cs = estimate_chunksize(a)
  GridChunks(a,cs)
end

struct Chunked end
struct Unchunked end
function haschunks end
haschunks(x) = Unchunked()

"""
    element_size(a::AbstractArray)

Returns the approximate size of an element of a in bytes. This falls back to calling `sizeof` on 
the element type or to the value stored in `DiskArrays.fallback_element_size`. Methods can be added for 
custom containers. 
"""
function element_size(a::AbstractArray) 
  if isbitstype(eltype(a))
    return sizeof(eltype(a))
  elseif isbitstype(Base.nonmissingtype(eltype(a)))
    return sizeof(Base.nonmissingtype(eltype(a)))
  else
    @warn "Can not determine size of element type. Using DiskArrays.fallback_element_size[] = $(fallback_element_size[]) bytes"
    return fallback_element_size[]
  end
end

estimate_chunksize(a::AbstractArray) = estimate_chunksize(size(a), element_size(a))
function estimate_chunksize(s, si)
  ii = searchsortedfirst(cumprod(collect(s)),default_chunk_size[]*1e6/si)
  ntuple(length(s)) do idim
    if idim<ii
      return s[idim]
    elseif idim>ii
      return 1
    else
      sbefore = idim == 1 ? 1 : prod(s[1:idim-1])
      return floor(Int,default_chunk_size[]*1e6/si/sbefore)
    end
  end
end
