#
# Date created: 2022-06-15
# Author: aradclif
#
#
############################################################################################

_typeofinv(x) = typeof(inv(x))

"""
    normalize1!(A::AbstractArray{<:Real})

Normalize the values in `A` such that `sum(A) ≈ 1` and `0 ≤ A[i] ≤ 1` ∀i.
It is assumed that `A[i] ≥ 0 ∀i`. This is not quite the L¹-norm, which would
require that `abs(A[i])` be used.

See also: [`normalize1`](@ref)
"""
function normalize1!(A::AbstractArray{T}) where {T<:Real}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @inbounds @simd for i ∈ eachindex(A)
        A[i] *= c
    end
    A
end

"""
    normalize1!(B::AbstractArray{<:Real}, A::AbstractArray{<:Real})

Normalize the values in `A` such that `sum(B) ≈ 1` and `0 ≤ B[i] ≤ 1` ∀i, storing
the result in `B`. It is assumed that `A[i] ≥ 0` ∀i.
"""
function normalize1!(B::AbstractArray{T}, A::AbstractArray{S}) where {T<:Real, S<:Real}
    s = zero(S)
    @inbounds @simd for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @inbounds @simd for i ∈ eachindex(A, B)
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
normalize1(A::AbstractArray{T<:Real}) = normalize1!(similar(A, _typeofinv(first(A))), A)

################
@noinline function _check_normweights(I::Vector{Int}, x)
    mn, mx = extrema(I)
    f, l = firstindex(x), lastindex(x)
    mn ≥ f || throw(BoundsError(x, mn))
    mx ≤ l || throw(BoundsError(x, mx))
end

#### Algorithm 2.1.
# I ∈ ℕᴺ, 𝐰 ∈ ℝᴰ; I ⊆ 1,…,D
# -> ω ∈ ℝᴺ, ωᵢ = 𝐰ᵢ / ∑ⱼ 𝐰ⱼ; j ∈ I

function unsafe_normweights!(p::Vector{T}, I::Vector{Int}, w::Vector{S}) where {T<:Real, S<:Real}
    s = zero(S)
    @inbounds @simd for i ∈ eachindex(I, p)
        w̃ = w[I[i]]
        s += w̃
        p[i] = w̃
    end
    c = inv(s)
    @inbounds @simd for i ∈ eachindex(p)
        p[i] *= c
    end
    return p
end

"""
    normweights!(p::Vector{T}, I::Vector{Int}, w::Vector{S}) where {T<:Real, S<:Real}

Fill `p` with the probabilities that result from normalizing the weights selected by `I` from `w`.
Note that `T` must be a type which is able to hold the result of `inv(one(S))`.
"""
function normweights!(p::Vector{T}, I::Vector{Int}, w::Vector{S}) where {T<:Real, S<:Real}
    _check_normweights(I, w)
    unsafe_normweights!(p, I, w)
end

"""
    normweights(I::Vector{Int}, w::Vector{<:Real})

Create a vector of probabilities by normalizing the weights selected by `I` from `w`.

See also: [`normweights!`](@ref)
"""
normweights(I::Vector{Int}, w::Vector{<:Real}) = normweights!(similar(I, _typeofinv(first(w))), I, w)
# Or, just promote_type(T, Float64)?



################

#### Algorithm 2.1. + Algorithm 3. (fused)

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





# A weight is assigned to each i, and the w₁'s are normalized to probabilities.
# Then, a subset of the i's, denoted I′, is selected for re-weighting by a quantity
# which is undefined for I ∖ I′.
# w₁ ∈ ℝᴰ : the weight assigned to each i for the normalization of probabilities
# w₂ ∈ ℝᴺ : the quantity which is undefined for I ∖ I′; undefined shall be encoded
# by a value of zero.
# pᵢ = w₁ᵢ / ∑ₗ₌₁ᴺ w₁ₗ, i ∈ I ∖ I′
# mᵏⁿᵒʷⁿ = ∑ᵢ pᵢ, i ∈ ∈ I ∖ I′
# mᵘⁿᵈᵉᶠⁱⁿᵉᵈ = 1 - mᵏⁿᵒʷⁿ = (∑ᵢ w₁ᵢ, i ∈ I′) / ∑ₗ₌₁ᴺ w₁ₗ
# pᵢ = mᵘⁿᵈᵉᶠⁱⁿᵉᵈ * w₂ᵢ / ∑ₗ₌₁ᴺ w₂ₗ, i ∈ I′
# In other words,
# pᵢ = (w₂ᵢ * ∑ₗ w₁ₗ, i ∈ I ∖ I′) / (∑ₗ₌₁ᴺ w₂ₗ * ∑ₗ₌₁ᴺ w₁ₗ)

function reweight!(w₁::Vector{T}, w₂::Vector{U}) where {T<:AbstractFloat, U<:Real}
    s₁′ = zero(T)
    s₁ = zero(T)
    s₂ = zero(U)
    @inbounds @simd for i ∈ eachindex(w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        s₁′ += w₂ᵢ == zero(U) ? zero(T) : w₁ᵢ
        s₁ += w₁ᵢ
        s₂ += w₂ᵢ
    end
    c₁ = s₁ == zero(T) ? one(T) : inv(s₁)
    c₂ = s₁′ * c₁ / s₂
    @inbounds @simd for i ∈ eachindex(w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        w₁[i] = w₂ᵢ == zero(U) ? c₁ * w₁ᵢ : c₂ * w₂ᵢ
    end
    w₁
end

function reweight!(p::Vector{S}, w₁::Vector{T}, w₂::Vector{U}) where {S<:AbstractFloat, T<:Real, U<:Real}
    s₁′ = zero(T)
    s₁ = zero(T)
    s₂ = zero(U)
    @inbounds @simd for i ∈ eachindex(w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        s₁′ += w₂ᵢ == zero(U) ? zero(T) : w₁ᵢ
        s₁ += w₁ᵢ
        s₂ += w₂ᵢ
    end
    # This covers an odd case, wherein w₁ consists of all zeros.
    # Naturally, there is not a clear definition for what the resultant probabilities
    # should be -- all zero, or 1/length(w₁)? this leans in favor of all zero.
    # s₁ = s₁ == zero(T) ? one(T) : s₁
    # c₁ = inv(s₁)
    # c₂ = s₁′ / (s₁ * s₂)
    c₁ = s₁ == zero(T) ? one(T) : inv(s₁)
    c₂ = s₁′ * c₁ / s₂
    @inbounds @simd for i ∈ eachindex(w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        p[i] = w₂ᵢ == zero(U) ? c₁ * w₁ᵢ : c₂ * w₂ᵢ
    end
    p
end

reweight(w₁::Vector{T}, w₂::Vector{U}) where {T<:Real, U<:Real} =
    reweight!(similar(w₁, promote_type(T, U, Float64)), w₁, w₂)
