#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################
# An efficient implementation of the Bernoulli process for sampling from a categorical
# distribution.
#### Bernoulli process
# θ ~ Multin(1; p₁, …, pₖ)
# θ₁ ~ Bin(1, p₁)
# θ₂, …, θₖ₋₁ ~ Bin(1 - ∑ᵢ₌₁ʲ⁻¹θᵢ, pⱼ / ∑ᵢ₌ⱼᵏpᵢ)
# θₖ = 1 - ∑ᵢ₌₁ᵏ⁻¹θᵢ
# Implies
# θⱼ ~ Bin(1 - ∑ᵢ₌₁ʲ⁻¹θᵢ, pⱼ / ∑ᵢ₌ⱼᵏ)
# In essence, the process is defined by repeated Bernoulli trials of
#     j         ∨     rest
# pⱼ / ∑ᵢ₌ⱼᵏpᵢ     1 - pⱼ / ∑ᵢ₌ⱼᵏpᵢ
# The process ends if j occurs at any point
#### Alternate phrasing
# Consider the remaining probability mass
#     ∑ᵢ₌ⱼᵏpᵢ = 1       - ∑ᵢ₌₁ʲ⁻¹pᵢ
#               ∑ᵢ₌₁ᵏpᵢ - ∑ᵢ₌₁ʲ⁻¹pᵢ
# ∑ᵢ₌₁ʲ⁻¹pᵢ is the probability mass removed when one is on the jᵗʰ category
# Hence, the process can be viewed as
#         j             ∨          rest
# pⱼ / (1 - ∑ᵢ₌₁ʲ⁻¹pᵢ)     1 - pⱼ / (1 - ∑ᵢ₌₁ʲ⁻¹pᵢ)
####
# The alternate phrasing of the Bernoulli process enables efficient sampling by
# avoiding the cumulative sums which would otherwise be required. The real
# advantages are only realized during the analogous Binomial process (which samples
# from the Multinomial distribution with n > 1). While the Bernoulli process
# provides a clever general approach to constructing reversible Markov
# transition kernels (involving the categorical distribution),
# there exists a faster sampler for categorical distributions in which all
# pⱼ's are known. In essence, one can utilize the properties of the cumulative
# distribution function, CDF(p, j) = ∑ᵢ₌₁ʲpᵢ, noting that a sample
# u ~ U(0, 1) belongs to the jᵗʰ category where j is the first
# j which satisfies CDF(p, j) > u. (In other words, one notes that the sample
# u could not have fallen into the j-1ᵗʰ category as ∑ᵢ₌₁ʲ⁻¹pᵢ < u).
################################################################
# using BenchmarkTools, Statistics


# As a recursion
# function catrand(p::Vector{T}, j::Int, k::Int, d::Float64=1.0) where {T<:Real}
#     if j ≤ k - 1
#         u = rand()
#         pⱼ = p[j]
#         p′ = pⱼ / d
#         return u ≤ p′ ? j : catrand(p, j + 1, k, d - pⱼ)
#     else
#         return j
#     end
# end
# catrand(p::Vector{T}) where {T<:Real} = catrand(p, 1, length(p), 1.0)
# catrandmulti₀(p::Vector{T}, N::Int) where {T<:Real} = [catrand(p) for _ = 1:N]

# p = [0.1, 0.25, 0.05, 0.35, 0.25]

# @benchmark catrand($p, 1, 5, 1.0)
# @benchmark catrand($p)
# @benchmark catrandmulti₀($p, 1000000)
# @timev c = catrandmulti₀(p, 1000000);
# t = [count(==(i), c) for i = 1:5]

# function rmass(p::Vector{T}, j::Int, i::Int=1, d::Float64=1.0) where {T<:Real}
#     if j == 1
#         return d
#     elseif i ≤ j - 1
#         return rmass(p, j, i + 1, d - p[i])
#     else
#         return d
#     end
# end

# function bernoullimass(p::Vector{T}) where {T<:Real}
#     k = length(p)
#     d = one(promote_type(T, Float64))
#     p̃ = Vector{promote_type(T, Float64)}(undef, k)
#     for j ∈ eachindex(p, p̃)
#         pⱼ = p[j]
#         p̃[j] = pⱼ / d
#         d -= pⱼ
#     end
#     return p̃
# end

# rmass(p, 2)
# r = map(j -> rmass(p, j), 1:5)
# @benchmark bernoullimass(p)
# p̃ = bernoullimass(p)
# p̃ == p ./ r

# If one wanted to repeat the computation several times, one could try:
# function catrandmultiₗ(p::Vector{T}, N::Int) where {T<:Real}
#     p̃ = bernoullimass(p)
#     k = length(p)
#     c = Vector{Int}(undef, N)
#     for n ∈ eachindex(c)
#         for j ∈ eachindex(p̃)
#             rand() ≤ p̃[j] && (c[n] = j; break)
#         end
#     end
#     return c
# end

# @benchmark catrandmultiₗ($p, 1000000)
# c2 = catrandmultiₗ(p, 1000000);
# t2 = [count(==(i), c2) for i = 1:5]

#### Essentially, there is very little difference in cost between rcatrand using
# pre-computed p̃ and dynamically computing it recursively. Correspondingly,
# the looped catrandmulti is faster than the recursive one.
# Particularly in the case of the loop it makes sense -- the resultant
# machine code is just much simpler.
# function rcatrand(p̃::Vector{T}, j::Int, k::Int) where {T<:Real}
#     if j ≤ k - 1
#         u = rand()
#         return u ≤ p̃[j] ? j : rcatrand(p̃, j + 1, k)
#     else
#         return j
#     end
# end
# rcatrand(p̃::Vector{T}) where {T<:Real} = rcatrand(p̃, 1, length(p̃))

# @benchmark rcatrand($p̃)
# @timev cᵣ = [rcatrand(p̃) for _ = 1:1000000];
# tᵣ = [count(==(i), cᵣ) for i = 1:5]

# function catrandmultiᵣ(p::Vector{T}, N::Int) where {T<:Real}
#     p̃ = bernoullimass(p)
#     k = length(p)
#     c = Vector{Int}(undef, N)
#     for n ∈ eachindex(c)
#         c[n] = rcatrand(p̃, 1, k)
#     end
#     return c
# end

# @benchmark catrandmultiᵣ($p, 1000000)

#### Attempt using a loop for the catrand itself, rather than recursion.
# For larger k's, the loop really pulls ahead -- at k = 100, it is twice as fast.
# At k = 1000, twice as fast.
# In essence, while the recursion is conceptually elegant, a loop provides a simpler interface
# AND 2x speed.
# function catrandₗ(p::Vector{T}) where {T<:Real}
#     k = length(p)
#     d = one(promote_type(T, Float64))
#     for j ∈ eachindex(p)
#         pⱼ = p[j]
#         u = rand()
#         u ≤ pⱼ / d && return j
#         d -= pⱼ
#     end
#     return k
# end


# @benchmark catrandₗ($p)
# @benchmark catrand($p)

# @timev cₗ = [catrandₗ(p) for _ = 1:1000000];
# tₗ = [count(==(i), cₗ) for i = 1:5]

# function norm1!(w::Vector{T}) where {T<:Real}
#     s = zero(T)
#     @inbounds @simd for i ∈ eachindex(w)
#         s += w[i]
#     end
#     c = inv(s)
#     @inbounds @simd for i ∈ eachindex(w)
#         w[i] *= c
#     end
#     return w
# end

# p2 = norm1!(rand(100));
# @benchmark catrandₗ($p2)
# @benchmark catrand($p2)

# p3 = norm1!(rand(1000));
# @benchmark catrandₗ($p3)
# @benchmark catrand($p3)

# p4 = norm1!(rand(10000));
# @benchmark catrandₗ($p4)
# @benchmark catrand($p4)

# # If one wanted to repeat the computation several times, one could try:
# function catrandₗ(p::Vector{T}, dims::Vararg{Int, N}) where {T<:Real} where {N}
#     p̃ = bernoullimass(p)
#     c = Array{Int, N}(undef, dims)
#     for n ∈ eachindex(c)
#         for j ∈ eachindex(p̃)
#             rand() ≤ p̃[j] && (c[n] = j; break)
#         end
#     end
#     return c
# end
# @benchmark catrandmultiₗ($p, 10000)
# @benchmark catrandₗ($p, 10000)

# using Distributions
# @benchmark rand(Categorical($p), 10000)
# c = rand(Categorical(p), 1000000);
# t = [count(==(i), c) for i = 1:5]

# #### Attempt using meta-unrolling -- 10x worse! (unless Val(k) is provided directly)
# @generated function catrandₘ(p::Vector{T}, ::Val{k}) where {T<:Real} where {k}
#     quote
#         d = one(T)
#         Base.Cartesian.@nexprs $k j -> rand() ≤ p[j] / d ? (return j) : d -= p[j]
#         return $k
#     end
# end
# catrandₘ(p::Vector{T}) where {T<:Real} = catrandₘ(p, Val(length(p)))

# @timev cₘ = [catrandₘ(p) for _ = 1:1000000];
# tₘ = [count(==(i), cₘ) for i = 1:5]
# @benchmark catrandₘ($p, Val(5))
# @benchmark catrandₘ($p)

#### A revised sampler using CDF properties -- improves speed by ≈1.3 at small k,
# and at large k, is considerably faster.

"""
    categorical(p::Vector{<:Real})

Draw a sample from the categorical distribution, where the number
of categories is equal to the length of `p`. Caller is responsible for ensuring that `∑p = 1`.

See also: [`categorical!`](@ref)
"""
@inline function categorical(p::AbstractVector{T}) where {T<:Real}
    k = length(p)
    j = 1
    s = p[1]
    u = rand()
    @inbounds while s < u && j < k
        s += p[j += 1]
    end
    return j
end

# @inline function categorical(p::AbstractVector{T}, Iₛ::Vector{Int}) where {T<:Real}
#     k = length(p)
#     j = 1
#     s = p[1]
#     u = rand()
#     @inbounds while s < u && j < k
#         s += p[j += 1]
#     end
#     return Iₛ[j]
# end


"""
    categorical!(C::Array{<:Integer, N}, p::Vector{<:Real}) where {N}

Fill `C` with draws from the k-dimensional categorical distribution defined by
the vector of probabilities `p`. The time complexity of this call should be assumed
to be greater than the batch method, as `rng` internal calls are sequential.
This may be useful when the memory overhead of a batch `rng` call exceeds the
time savings.
Caller is responsible for ensuring that `∑p = 1`.
"""
@inline function categorical!(C::AbstractArray{S, N}, p::AbstractVector{T}) where {T<:Real} where {S<:Integer, N}
    k = length(p)
    Σp = cumsum(p)
    @inbounds for i ∈ eachindex(C)
        j = 1
        s = Σp[1]
        u = rand()
        while s < u && j < k
            s = Σp[j += 1]
        end
        C[i] = j
    end
    return C
end

"""
    categorical(p::Vector{<:Real}, dims::Int...)

Sample an array of categories from the k-dimensional categorical distribution
defined by the vector of probabilities `p`.
"""
@inline categorical(p::AbstractVector{T}, dims::Vararg{Int, N}) where {T<:Real} where {N} =
    categorical!(Array{Int, N}(undef, dims), p)

############################################################################################
#### 2022-04-11: Batch SIMD sampler

"""
    categorical!(C::Array{<:Integer, N}, U::Array{T, N}, Σp::Vector{T}) where {T<:Real, N}

Fill `C` with draws from the k-dimensional categorical distribution defined by
the vector of cumulative probabilities `Σp`. This method is optimized
(internal `rng` calls are batched) for repeated calls involving arrays `C`, `U`
of the same size, potentially with different `Σp`'s.

Note: `U` is storage, potentially uninitialized, for the uniform random draws
which will ultimately be used to draw samples from the categorical distribution.
"""
@inline function categorical!(C::AbstractArray{S, N}, U::AbstractArray{T, N}, Σp::AbstractVector{T}) where {T<:AbstractFloat} where {N} where {S<:Integer}
    k = length(Σp)
    rand!(U)
    @inbounds for i ∈ eachindex(C, U)
        u = U[i]
        j = 1
        s = Σp[1]
        while s < u && j < k
            s = Σp[j += 1]
        end
        C[i] = j
    end
    return C
end

"""
    categorical!(C::Array{<:Integer, N}, U::Array{T, N}, Σp::Vector{T}, p::Vector{T}) where {T<:AbstractFloat, N}

`Σp` is a vector of any size, potentially uninitialized, which will be
`resize!`'d and filled with the cumulative probabilities required for sampling.
"""
@inline function categorical!(C::AbstractArray{S, N}, U::AbstractArray{T, N}, Σp::AbstractVector{T}, p::AbstractVector{T}) where {T<:AbstractFloat} where {N} where {S<:Integer}
    k = length(p)
    resize!(Σp, k)
    cumsum!(Σp, p)
    rand!(U)
    @inbounds for i ∈ eachindex(C, U)
        u = U[i]
        j = 1
        s = Σp[1]
        while s < u && j < k
            s = Σp[j += 1]
        end
        C[i] = j
    end
    return C
end


############################################################################################
#### 2022-05-20: Multinomial
# # Demonstrate that the sum of categorical distributions cannot be represented by
# # a multinomial distribution.

# # Test 1
# Is = [[1,2], [1,2,3,4], [1,2,3,4,5,6]];
# A = sample(Int, Is, 6, 10^6);
# ω = A ./ 3;
# mean(ω, dims=2)
# var(ω, dims=2)

# # Property tests
# # We know that the moments and properties will all be correct, as this is the only correct
# # way to simulate the distribution of θ₁ + θ₂ + θ₃
# mean(A, dims=2)
# # Var(θ₁ + θ₂ + θ₃) = 1/4 + 3/16 + 5/36
# var(A, dims=2)
# # Pr(θ₁ = 1 ∩ θ₂ = 1 ∩ θ₃ = 1) = 1/48
# count(==(3), A, dims=2) ./ 10^6


# ws = [fill(1/2, 2), fill(1/4, 4), fill(1/6, 6)];
# function multin(ws::Vector{Vector{T}}) where {T<:Real}
#     ω = zeros(promote_type(T, Float64), maximum(length, ws))
#     for w ∈ ws
#         for i ∈ eachindex(w)
#             ω[i] += w[i]
#         end
#     end
#     J⁻¹ = one(promote_type(T, Float64)) / length(ws)
#     for i ∈ eachindex(ω)
#         ω[i] *= J⁻¹
#     end
#     ω
# end

# # ws′ = [[w; zeros(6 - length(w))] for w ∈ ws]
# # ws′[1] * ws′[2]' .* reshape(ws′[3], 1,1,6)
# # ws[1] * transpose(ws[2]) .* reshape(ws[3], 1,1,6)

# # # Test 2
# # w = rand(6)
# # ws₂ = [normweights(I, w) for I ∈ Is]
# # A₂ = sample(Int, Is, w, 6, 10^6);
# # ω₂ = A₂ ./ 3;
# # mean(ω₂, dims=2)
# # var(ω₂, dims=2)

# # ω₂′ = multin(ws₂)

# # w′ = normalize1(w)
# # [normweights(I, w′) for I ∈ Is]

# using Distributions, BenchmarkTools
# ω′ = multin(ws)
# d = Multinomial(3, ω′);
# # Multinomial is approximately 10x slower than Bernoulli process, though, this is small N.
# # I would expect the Bernoulli-Binomial process (presumably) used in the Multinomial
# # to become faster at larger N
# # @benchmark rand(d, 100000)
# # @benchmark sample(Int, Is, 6, 100000)
# A₃ = rand(d, 10^6);
# ω₃ = A₃ ./ 3;
# mean(ω₃, dims=2)
# var(ω₃, dims=2)
# 11/36

# # Property tests
# # We know the expectation will match, but the variance will not be correct,
# # nor will other properties such conditional probabilities
# mean(A₃, dims=2)
# # Var(θ₁ + θ₂ + θ₃) = 1/4 + 3/16 + 5/36
# var(A₃, dims=2)
# # Pr(θ₁ = 1 ∩ θ₂ = 1 ∩ θ₃ = 1) = 1/48, but clearly not true, as shown below
# count(==(3), A₃, dims=2) ./ 10^6
