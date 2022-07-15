#
# Date created: 2022-06-13
# Author: aradclif
#
#
############################################################################################
# Conveniences
const _mni64 = typemin(Int)
const _mxi64 = typemax(Int)

_maximum_maybe(x::AbstractVector{T}) where {T<:Integer} = isempty(x) ? zero(T) : maximum(x)
_maximum_maybe((x, y)::Tuple{Vector{Int}, Vector{T}}) where {T<:AbstractFloat} = _maximum_maybe(x)

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M} =
    isempty(A) ? 0 : maximum(num_cat, A, init=_mni64)
num_cat(A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {T<:AbstractFloat, N} =
    isempty(A) ? 0 : maximum(_maximum_maybe, A, init=_mni64)
num_cat(A::Tuple{Vector{Int}, Vector{T}}) where {T<:AbstractFloat} = _maximum_maybe(A)

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{Int}, M}, N} where {M} =
    isempty(A) ? 0 : maximum(num_cat, A, init=_mni64)
num_cat(A::AbstractArray{Vector{Int}, N}) where {N} =
    isempty(A) ? 0 : maximum(_maximum_maybe, A, init=_mni64)
num_cat(A::Vector{Int}) = _maximum_maybe(A)

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M} =
    isempty(A) ? 0 : maximum(num_cat, A, init=_mni64)
num_cat(A::AbstractArray{Vector{T}, N}) where {T<:AbstractFloat, N} =
    isempty(A) ? 0 : maximum(length, A, init=_mni64)
num_cat(A::Vector{T}) where {T<:AbstractFloat} = length(A)

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M} =
    isempty(A) ? 0 : maximum(num_cat, A, init=_mni64)
num_cat(A::AbstractArray{SparseVector{Tv, Ti}, N}) where {Tv<:AbstractFloat, Ti<:Integer, N} =
    isempty(A) ? 0 : maximum(length, A, init=_mni64)
num_cat(A::SparseVector{Tv, Ti}) where {Tv<:AbstractFloat, Ti<:Integer} = length(A)

_minimum_maybe(x::AbstractVector{T}) where {T<:Integer} = isempty(x) ? one(T) : minimum(x)
_minimum_maybe((x, y)::Tuple{Vector{Int}, Vector{T}}) where {T<:AbstractFloat} = _minimum_maybe(x)

num_cat_min(A::AbstractArray{R, N}) where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M} =
    isempty(A) ? 1 : minimum(num_cat_min, A, init=_mxi64)
num_cat_min(A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {T<:AbstractFloat, N} =
    isempty(A) ? 1 : minimum(_minimum_maybe, A, init=_mxi64)
num_cat_min(A::Tuple{Vector{Int}, Vector{T}}) where {T<:AbstractFloat} = _minimum_maybe(A)

num_cat_min(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{Int}, M}, N} where {M} =
    isempty(A) ? 1 : minimum(num_cat_min, A, init=_mxi64)
num_cat_min(A::AbstractArray{Vector{Int}, N}) where {N} =
    isempty(A) ? 1 : minimum(_minimum_maybe, A, init=_mxi64)
num_cat_min(A::Vector{Int}) = _minimum_maybe(A)

num_cat_min(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M} = 1
num_cat_min(A::AbstractArray{Vector{T}, N}) where {T<:AbstractFloat, N} = 1
num_cat_min(A::Vector{T}) where {T<:AbstractFloat} = 1

num_cat_min(A::AbstractArray{R, N}) where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M} = 1
num_cat_min(A::AbstractArray{SparseVector{Tv, Ti}, N}) where {Tv<:AbstractFloat, Ti<:Integer, N} = 1
num_cat_min(A::SparseVector{Tv, Ti}) where {Tv<:AbstractFloat, Ti<:Integer} = 1

####
# _extrema_maybe(x::AbstractVector{T}) where {T<:Real} = isempty(x) ? (zero(T), zero(T)) : extrema(x)

# bounds_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M} =
#     (((lb1, ub1), (lb2, ub2)) = extrema(bounds_cat, A, init=((1,1), (0,0)));
#      extrema((lb1, ub1, lb2, ub2)))
# bounds_cat(A::AbstractArray{T, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}}, N} =
#     (((lb1, ub1), (lb2, ub2)) = extrema(bounds_cat, A, init=((1,1), (0,0)));
#      extrema((lb1, ub1, lb2, ub2)))
# bounds_cat(A::Tuple{Vector{Int}, Vector{T}}) where {T<:AbstractFloat} = _extrema_maybe(A[1])

# bounds_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{Int}, M}, N} where {M} =
#     (((lb1, ub1), (lb2, ub2)) = extrema(bounds_cat, A, init=((1,1), (0,0)));
#      extrema((lb1, ub1, lb2, ub2)))
# bounds_cat(A::AbstractArray{Vector{Int}, N}) where {N} =
#     (((lb1, ub1), (lb2, ub2)) = extrema(_extrema_maybe, A, init=((1,1), (0,0)));
#      extrema((lb1, ub1, lb2, ub2)))
# bounds_cat(A::Vector{Int}) = _extrema_maybe(A)

# bounds_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M} = (n = num_cat(A); n ≥ 1 ? (1, n) : (0, 0))
# bounds_cat(A::AbstractArray{Vector{T}, N}) where {T<:AbstractFloat, N} = (n = num_cat(A); n ≥ 1 ? (1, n) : (0, 0))
# bounds_cat(A::Vector{T}) where {T<:AbstractFloat} = (n = num_cat(A); n ≥ 1 ? (1, n) : (0, 0))

# bounds_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M} =
#     (n = num_cat(A); n ≥ 1 ? (1, n) : (0, 0))
# bounds_cat(A::AbstractArray{SparseVector{Tv, Ti}, N}) where {Tv<:AbstractFloat, Ti<:Integer, N} =
#     (n = num_cat(A); n ≥ 1 ? (1, n) : (0, 0))
# bounds_cat(A::SparseVector{T}) where {T<:AbstractFloat} = (n = num_cat(A); n ≥ 1 ? (1, n) : (0, 0))

#
bounds_cat(x::Int, y::Int) = (x, y)
bounds_cat(A::T) where {T} = bounds_cat(num_cat_min(A), num_cat(A))
################
function _checkindex_reducedims(ax, lb, ub)
    (checkindex(Bool, ax, lb) && checkindex(Bool, ax, ub)) || throw(DimensionMismatch("cannot sample from categories on range $(lb:ub) into array with first dimension $(ax)"))
    true
end

@noinline function _check_reducedims(B, A)
    Rdims = axes(B)[3:end]
    lb, ub = bounds_cat(A)
    _checkindex_reducedims(axes(B, 2), lb, ub)
    length(Rdims) ≤ ndims(A) || throw(DimensionMismatch("cannot reduce $(ndims(A))-dimensional array to $(length(Rdims)) trailing dimensions"))
    for i ∈ eachindex(Rdims)
        Ri, Ai = Rdims[i], axes(A, i)
        length(Ri) == 1 || Ri == Ai || throw(DimensionMismatch("reduction on array with indices $(axes(A)) with output with indices $(Rdims)"))
    end
    true
end

for Tₐ ∈ (Tuple{Vector{Int}, Vector{<:AbstractFloat}}, Vector{Int}, Vector{<:AbstractFloat}, SparseVector{<:AbstractFloat})
    @eval @noinline function _check_reducedims(B, A::$Tₐ)
        lb, ub = bounds_cat(A)
        _checkindex_reducedims(axes(B, 2), lb, ub)
        true
    end
end

################################################################
"""
    splitranges(start, stop, chunksize)

Divide the range `start:stop` into segments, each of size `chunksize`.
The last segment will contain the remainder, `(start - stop + 1) % chunksize`,
if it exists.
"""
function splitranges(start::Int, stop::Int, Lc::Int)
    L = stop - start + 1
    n, r = divrem(L, Lc)
    ranges = Vector{UnitRange{Int}}(undef, r == 0 ? n : n + 1)
    l = start
    @inbounds for i = 1:n
        l′ = l
        l += Lc
        ranges[i] = l′:(l - 1)
    end
    if r != 0
        @inbounds ranges[n + 1] = (stop - r + 1):stop
    end
    return ranges
end

"""
    splitranges(ur::UnitRange{Int}, chunksize)

Divide the range `ur` into segments, each of size `chunksize`.
"""
splitranges(ur::UnitRange{Int}, Lc::Int) = splitranges(ur.start, ur.stop, Lc)

################################################################
# sampler accessories
_largesmall_init(n::Int) = Vector{Int}(undef, n), Vector{Int}(undef, n)
@inline _genstorage_init(T::Type{<:AbstractFloat}, n::Int) = Vector{Int}(undef, n), Vector{T}(undef, n)
