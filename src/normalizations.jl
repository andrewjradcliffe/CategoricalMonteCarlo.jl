#
# Date created: 2022-06-15
# Author: aradclif
#
#
############################################################################################

"""
    normweights!(p::Vector{S}, w::Vector{T}, I::Vector{Int}) where {T<:Real, S<:AbstractFloat}

Fill `p` with the probabilities that result from normalizing the weights selected by `I` from `w`.
"""
function normweights!(p::Vector{S}, I::Vector{Int}, w::Vector{T}) where {T<:Real, S<:AbstractFloat}
    s = zero(T)
    @inbounds @simd ivdep for i ∈ eachindex(I, p)
        w̃ = w[I[i]]
        s += w̃
        p[i] = w̃
    end
    c = inv(s)
    @inbounds @simd ivdep for i ∈ eachindex(p)
        p[i] *= c
    end
    return p
end

"""
    normweights(w::Vector{<:Real}, I::Vector{Int})

Create a vector of probabilities by normalizing the weights selected by `I` from `w`.

See also: [`normweights!`](@ref)
"""
normweights(I::Vector{Int}, w::Vector{T}) where {T<:Real} =
    normweights!(similar(I, promote_type(T, Float64)), I, w)


# A weight is assigned to i = 1,…,k components, and there are unknown components k+1,…,N.
# The unknown components are of the same category, and the probability mass of the category is
# known; alternatively, the ratio (between unknown/known) of probability masses may be specified.
# r = unknown/known = (∑ᵢ₌ₖ₊₁ᴺ pᵢ) / ∑ᵢ₌₁ᵏ pᵢ = (∑ᵢ₌ₖ₊₁ᴺ wᵢ) / ∑ᵢ₌₁ᵏ wᵢ ⟹
# r∑ᵢ₌₁ᵏ wᵢ = ∑ᵢ₌ₖ₊₁ᴺ wᵢ ⟹ r∑ᵢ₌₁ᵏ = w′, wᵢ = w′ / (N - k), i=k+1,…,N
# r = u / (1 - u) ⟹ u = r / (1 + r) ⟹
# pᵢ = u / (N - k), i=k+1,…,N
# pᵢ = (1 - u) wᵢ / ∑ᵢ₌₁ᵏ wᵢ, i = 1,…,k
"""
    normweights!(p::Vector{S}, I::Vector{Int}, w::Vector{T}, u::S) where {S<:AbstractFloat, T<:Real}

Fill `p` with the probabilities which result from normalizing the weights selected by `I`
from `w`, wherein zero or more of the elements of `w` has an unknown (set to 0) value.
The total probability mass of the unknown category is specified by `u`.
Caller must ensure that `u` is in the closed interval [0, 1].
If all selected values are zero, `p` is filled with `1 / length(p)`.

See also: [`normweights`](@ref)
"""
function normweights!(p::Vector{S}, I::Vector{Int}, w::Vector{T}, u::S) where {S<:AbstractFloat, T<:Real}
    s = zero(T)
    z = 0
    @inbounds @simd for i ∈ eachindex(p, I)
        w̃ = w[I[i]]
        s += w̃
        p[i] = w̃
        z += w̃ == zero(T)
    end
    c = z == 0 ? inv(s) : (one(S) - u) / s
    u′ = z == length(p) ? inv(z) : u / z
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = pᵢ == zero(S) ? u′ : pᵢ * c
        # p[i] = ifelse(pᵢ == zero(S), u′, pᵢ * c)
    end
    p
end

"""
    normweights(I::Vector{Int}, w::Vector{<:Real}, u::AbstractFloat)

Return a vector of probabilities, selecting components from `w` using the index set `I`.
Categories with unknown weight are assumed to have a total probability mass `u`.

See also: [`normweights!`](@ref)
"""
normweights(I::Vector{Int}, w::Vector{T}, u::S) where {T<:Real, S<:AbstractFloat} =
    (zero(S) ≤ u ≤ one(S) || throw(DomainError(u)); normweights!(similar(I, promote_type(T, S, Float64)), I, w, u))

"""
    normweights!(p::Vector{S}, u::S) where {S<:AbstractFloat}

Normalize `p` to probabilities, spreading probability mass `u` across the
0 or more elements which have value(s) of zero. If all values of `p` are
equal to zero, `p` is filled with `1 / length(p)`.
"""
function normweights!(p::Vector{S}, u::S) where {S<:AbstractFloat}
    s = zero(S)
    z = 0
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        s += pᵢ
        z += pᵢ == zero(S)
    end
    c = z == 0 ? inv(s) : (one(S) - u) / s
    u′ = z == length(p) ? inv(z) : u / z
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = pᵢ == zero(S) ? u′ : pᵢ * c
    end
    p
end

"""
    normweights!(p::Vector{S}, w::Vector{<:Real}, u::S) where {S<:AbstractFloat}

Normalize `w` to probabilities, storing the result in `p`, spreading probability
mass `u` across the 0 or more elements which have value(s) of zero. If all values
of `w` are zero, `p` is filled with `1 / length(p)`.
"""
function normweights!(p::Vector{S}, w::Vector{T}, u::S) where {S<:AbstractFloat, T<:Real}
    s = zero(T)
    z = 0
    @inbounds @simd for i ∈ eachindex(p)
        w̃ = w[i]
        s += w̃
        p[i] = w̃
        z += w̃ == zero(T)
    end
    c = z == 0 ? inv(s) : (one(S) - u) / s
    u′ = z == length(p) ? inv(z) : u / z
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = pᵢ == zero(S) ? u′ : pᵢ * c
    end
    p
end

normweights(p::Vector{T}, u::S) where {T<:Real, S<:AbstractFloat} =
    normweights!(similar(p, promote_type(T, S)), p, u)

# This is not quite the L¹-norm, which would require that abs(A[i]) be used.
"""
    normalize1!(A::AbstractArray{<:Real})

Normalize the values in `A` such that `sum(A) ≈ 1` and `0 ≤ A[i] 1` ∀i.
It is assumed that `A[i] ≥ 0 ∀i`.

See also: [`normalize1`](@ref)
"""
function normalize1!(A::AbstractArray{T}) where {T<:Real}
    s = zero(promote_type(T, Float64))
    @inbounds @simd ivdep for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @inbounds @simd ivdep for i ∈ eachindex(A)
        A[i] *= c
    end
    A
end

"""
    normalize1!(B::AbstractArray{<:Real}, A::AbstractArray{<:Real})

Normalize the values in `A` such that `sum(B) ≈ 1` and `0 ≤ B[i] 1` ∀i, storing
the result in `B`. It is assumed that `A[i] ≥ 0` ∀i.
"""
function normalize1!(B::AbstractArray{T}, A::AbstractArray{S}) where {T<:AbstractFloat, S<:Real}
    s = zero(promote_type(T, Float64))
    @inbounds @simd ivdep for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @inbounds @simd ivdep for i ∈ eachindex(A, B)
        B[i] = A[i] * c
    end
    B
end

"""
    normalize1(A::AbstractArray{<:Real})

Return an array of equal size which satisfies `sum(B) ≈ 1` and `0 ≤ B[i] 1` ∀i.
It is assumed that `A[i] ≥ 0` ∀i.

See also: [`normalize1!`](@ref)
"""
normalize1(A::AbstractArray{T}) where {T<:Real} = normalize1!(similar(A, promote_type(T, Float64)), A)
