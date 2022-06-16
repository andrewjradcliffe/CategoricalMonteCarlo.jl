#
# Date created: 2022-06-13
# Author: aradclif
#
#
############################################################################################
# Conveniences
_maximum_maybe(x::AbstractVector{T}) where {T<:Real} = isempty(x) ? zero(T) : maximum(x)

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M} = maximum(a -> maximum(((I, w),) -> _maximum_maybe(I), a, init=0), A, init=0)
num_cat(A::AbstractArray{T, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}}, N} =
    maximum(((I, w),) -> _maximum_maybe(I), A, init=0)
num_cat(A::Tuple{Vector{Int}, Vector{T}}) where {T<:AbstractFloat} = _maximum_maybe(A[1])

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{Int}, M}, N} where {M} = maximum(a -> maximum(_maximum_maybe, a, init=0), A, init=0)
num_cat(A::AbstractArray{Vector{Int}, N}) where {N} = maximum(_maximum_maybe, A, init=0)
num_cat(A::Vector{Int}) = _maximum_maybe(A)

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M} = maximum(a -> maximum(length, a, init=0), A, init=0)
num_cat(A::AbstractArray{Vector{T}, N}) where {T<:AbstractFloat, N} = maximum(length, A, init=0)
num_cat(A::Vector{T}) where {T<:AbstractFloat} = length(A)

num_cat(A::AbstractArray{R, N}) where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M} = maximum(a -> maximum(length, a, init=0), A, init=0)
num_cat(A::AbstractArray{SparseVector{Tv, Ti}, N}) where {Tv<:AbstractFloat, Ti<:Integer, N} = maximum(length, A, init=0)
num_cat(A::SparseVector{T}) where {T<:AbstractFloat} = length(A)

@noinline function _check_reducedims(B, A)
    Rdims = axes(B)[3:end]
    n_cat = num_cat(A)
    n_cat′ = length(axes(B, 1))
    n_cat ≤ n_cat′ || throw(DimensionMismatch("cannot sample from $(n_cat) categories into array with $(n_cat′) categories"))
    length(Rdims) ≤ ndims(A) || throw(DimensionMismatch("cannot reduce $(ndims(A))-dimensional array to $(length(Rdims)) trailing dimensions"))
    for i ∈ eachindex(Rdims)
        Ri, Ai = Rdims[i], axes(A, i)
        length(Ri) == 1 || Ri == Ai || throw(DimensionMismatch("reduction on array with indices $(axes(A)) with output with indices $(Rdims)"))
    end
    true
end

@noinline function _check_reducedims(B, A::Tuple{Vector{Int}, Vector{<:AbstractFloat}})
    n_cat = num_cat(A)
    n_cat′ = length(axes(B, 1))
    n_cat ≤ n_cat′ || throw(DimensionMismatch("cannot sample from $(n_cat) categories into array with $(n_cat′) categories"))
    true
end

@noinline function _check_reducedims(B, A::Vector{Int})
    n_cat = num_cat(A)
    n_cat′ = length(axes(B, 1))
    n_cat ≤ n_cat′ || throw(DimensionMismatch("cannot sample from $(n_cat) categories into array with $(n_cat′) categories"))
    true
end

@noinline function _check_reducedims(B, A::Vector{T}) where {T<:AbstractFloat}
    n_cat = num_cat(A)
    n_cat′ = length(axes(B, 1))
    n_cat ≤ n_cat′ || throw(DimensionMismatch("cannot sample from $(n_cat) categories into array with $(n_cat′) categories"))
    true
end

@noinline function _check_reducedims(B, A::SparseVector{T}) where {T<:AbstractFloat}
    n_cat = num_cat(A)
    n_cat′ = length(axes(B, 1))
    n_cat ≤ n_cat′ || throw(DimensionMismatch("cannot sample from $(n_cat) categories into array with $(n_cat′) categories"))
    true
end
