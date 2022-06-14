#
# Date created: 2022-06-13
# Author: aradclif
#
#
############################################################################################
# Conveniences
_maximum_maybe(x::AbstractVector{T}) where {T<:Real} = isempty(x) ? zero(T) : maximum(x)

num_cat(A::AbstractArray{Vector{T}, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}}, N} =
    maximum(a -> maximum(((I, w),) -> _maximum_maybe(I), a, init=0), A, init=0)
num_cat(A::AbstractArray{T, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}}, N} =
    maximum(((I, w),) -> _maximum_maybe(I), A, init=0)
num_cat(A::Tuple{Vector{Int}, Vector{T}}) where {T<:AbstractFloat} = _maximum_maybe(A[1])

num_cat(A::AbstractArray{Vector{Vector{Int}}, N}) where {N} = maximum(a -> maximum(_maximum_maybe, a, init=0), A, init=0)
num_cat(A::AbstractArray{Vector{Int}, N}) where {N} = maximum(_maximum_maybe, A, init=0)
num_cat(A::Vector{Int}) = _maximum_maybe(A)

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
