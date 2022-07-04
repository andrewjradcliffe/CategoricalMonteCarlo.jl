#
# Date created: 2022-06-21
# Author: aradclif
#
#
############################################################################################
# https://www.jstatsoft.org/article/view/v011i03
#### Marsaglia's Square Histogram (Method II in the above article)
# p ∈ ℝᴺ, ∑ᵢpᵢ = 1, a = 1/N
# K ∈ ℕᴺ, Kᵢ = i
# V ∈ ℝⁿ, Vᵢ = i * a
# Generate: j = ⌊N*U+1⌋; if U < V[j], return j, else return K[j]
# Theoretically, just one U ~ Uniform(0,1) is sufficient. In practice, it is also faster as
#     U=rand(); j=floor(Int, N * U + 1)
# is far fewer instructions than
#     j=rand(1:N); U=rand()
#### Robin Hood
## Motivation
# The frequency of the `else` statement being required is proportional to the "over-area",
# i.e. the part of the final squared histogram that lies above the division points.
# Or, to quantify it: ∑ᵢ (i * a) - V[i]
# The objective is to minimize the number of times the `else return K[j]` occurs
# by creating V such that each V[j] is as close to j/N as possible -- an NP-hard problem.
# Marsaglia's suggestion of the Robin Hood method is a good solution which is 𝒪(NlogN).
## Thoughts
# The reduction in `else` statements leads to faster sampling -- 𝒪(1) regardless --
# as it would certainly lead to a more predictable instruction pipeline due to minimizing
# the number of times the `else` branch is executed (or used, even if executed).
# Does it justify the 𝒪(NlogN) construction cost for the alias tables? -- given
# that we could pay 𝒪(N) instead to construct an inferior table (but no limit on how terrible).
#     - An analysis based on the objective function above would be informative;
#       can be verified with Monte Carlo: compute 𝒻(𝐱) = ∑ᵢ (i * a) - V[i]
#       using the V produced by each procedure.
#     - Number of samples to be drawn is an orthogonal decision variable;
#       one surmises that increasing number of samples favor better table.
## Algorithm
# repeat these two steps N - 1 times
#     1. find the smallest probability, pᵢ, and the largest probability, pⱼ
#     2. set K[i] = j; V[i] = (i - 1) * a + pᵢ; replace pⱼ with pⱼ - (a - pᵢ)
## Numerical stability
# Replacing pⱼ is the only point at which stability is a real concern. There are a few
# options for the order of operations and parenthesis. Unsurprisingly, Marsaglia gives
# the most stable form: pⱼ = pⱼ - (a - pᵢ)
# But it is worthwhile to show that this is the most stable form.
#     First, consider that it should be the case that pᵢ ≤ a, hence 0 ≤ (a - pᵢ) ≤ 1/n.
#     (a - pᵢ) may be a small number, but (a - pᵢ) is always well-defined since it occurs
#     at eps(a)≡ulp(a).
# It is worth noting that the (a - pᵢ) operation becomes unstable when pᵢ ≤ eps(a)/4, assuming
# the worst case, a=1. However, this has the opposite relationship to n: increasing n will
# result in a subtraction which takes place at smaller values, hence, the (floating)
# points are more densely packed (i.e. distance to nearest float is smaller).
# It is reassuring to note that eps(.5) = 2⁻⁵³, hence, even for a vector of length 2,
# the (a - pᵢ) is stable for pᵢ > 2⁻⁵⁵.
#     The subsequent subtraction, i.e. pⱼ - (a - pᵢ), will occur at eps(pⱼ)≡ulp(pⱼ). Thus,
#     the operation will be unstable when pⱼ - c * ulp(pⱼ), c ≤ 1/4 (for pⱼ = 1, the worst case).
# That is, unstable when: (a - pᵢ) ≤ eps(pⱼ)/4
# If pᵢ ≈ 0, (a - pᵢ) ≈ 1/n ⟹ 1/n ≤ eps(pⱼ)/4 is unstable
# As pⱼ is at most 1, the worst case will be eps(pⱼ)/4 = 2⁻⁵⁴, i.e. 1/n ≤ 2⁻⁵⁴.
# ∴ in the worse case, instability begins at n ≥ 2⁵⁴ if pᵢ ≈ 0.
# ∴ in general, expect instability if (a - pᵢ) ≤ 2⁻⁵⁴.
# The above assumed Float64 with 53 bits of precision; the general form in terms of precision
# replaces 2⁻⁵⁴ → 2⁻ᵖ⁻¹.
# These are very permissive bounds; one is likely to run into other issues well before
# the algorithm becomes numerically unstable.
## Numerical stability, revisit
# Oddly, Marsaglia's implementation in TplusSQ.c uses pⱼ + pᵢ - a, which has slightly
# worse numerical stability. (pⱼ + pᵢ) becomes unstable when pᵢ ≤ eps(pⱼ)/2, which
# corresponds to 2⁻ᵖ at pⱼ = 1.
# It is possible to find cases where both suffer roundoff, which is ≤ 2⁻ᵖ⁺¹ for pⱼ + pᵢ - a
# and ≤ 2⁻ᵖ for pⱼ - (a - pᵢ).
# Provided that one is working with Float64, it most likely does not matter.
# However, if p is provided as Float32, it may be preferable to forcibly promote to Float64
# just to ensure stability; naturally, Float16 is largely unsuitable and needs promotion.
# ##
# # Comparison of stability
# # f1 is unstable at n = 10, qⱼ = .999999999999
# # However, f2 and f3 are unstable at n = 10, qⱼ = 5
# f1(qⱼ, qᵢ, a) = qⱼ - (a - qᵢ)
# f2(qⱼ, qᵢ, a) = qⱼ + qᵢ - a
# ff2(qⱼ, qᵢ, a) = @fastmath qⱼ + qᵢ - a
# f3(qⱼ, qᵢ, a) = (qⱼ + qᵢ) - a
# n = 10
# a = 1 / n
# qⱼ = .999999999999
# qᵢ = (1 - qⱼ) / n
# f1(qⱼ, qᵢ, a)
# f2(qⱼ, qᵢ, a)
# ff2(qⱼ, qᵢ, a)
# f3(qⱼ, qᵢ, a)
# f1(big(qⱼ), big(qᵢ), big(a))
# f2(big(qⱼ), big(qᵢ), big(a))
# f3(big(qⱼ), big(qᵢ), big(a))

function marsaglia(p::Vector{T}) where {T<:AbstractFloat}
    n = length(p)
    K = Vector{Int}(undef, n)
    V = Vector{promote_type(T, Float64)}(undef, n)
    q = similar(p, promote_type(T, Float64))
    a = inv(n)
    # initialize
    @inbounds for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:n-1
        qᵢ, i = findmin(q)
        qⱼ, j = findmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = qⱼ - (a - qᵢ)
        q[i] = a
    end
    K, V
end

function vmarsaglia(p::Vector{T}) where {T<:AbstractFloat}
    n = length(p)
    K = Vector{Int}(undef, n)
    V = Vector{promote_type(T, Float64)}(undef, n)
    a = inv(n)
    q = similar(p, promote_type(T, Float64))
    # initialize
    @turbo for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:n-1
        qᵢ, i = vfindmin(q)
        qⱼ, j = vfindmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = qⱼ - (a - qᵢ)
        q[i] = a
    end
    K, V
end

# p = [0.2, 0.3, 0.1, 0.4]

# p = [.21, .18, .26, .17, .18]

# p = [2/15, 7/15, 6/15]

function marsaglia_generate(K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    n = length(K)
    u = rand()
    j = trunc(Int, muladd(u, n, 1))
    u < V[j] ? j : K[j]
end

function marsaglia_generate!(A::AbstractArray, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    n = length(K)
    @inbounds for i ∈ eachindex(A) # safe to also use @fastmath, @simd
        u = rand()
        j = trunc(Int, muladd(u, n, 1)) # muladd is faster than u * n + 1 by ≈5-6%
        A[i] = u < V[j] ? j : K[j]
    end
    A
end
function marsaglia_generate!(A::AbstractArray, u::AbstractArray{Float64}, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    n = length(K)
    rand!(u)
    @inbounds for i ∈ eachindex(A, u) # safe to also use @fastmath, @simd
        j = trunc(Int, muladd(u[i], n, 1)) # muladd is faster than u * n + 1 by ≈5-6%
        A[i] = u[i] < V[j] ? j : K[j]
    end
    A
end

function marsaglia_generate(K::Vector{Int}, V::Vector{T}, dims::Vararg{Int, N}) where {T<:AbstractFloat} where {N}
    marsaglia_generate!(Array{Int}(undef, dims), K, V)
end

function marsaglia!(K::Vector{Int}, V::Vector{T}, q::Vector{T}, p::Vector{T}) where {T<:AbstractFloat}
    (length(K) == length(V) == length(q) == length(p)) || throw(ArgumentError("all inputs must be of same size"))
    n = length(p)
    a = inv(n)
    @inbounds for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:n-1
        qᵢ, i = findmin(q)
        qⱼ, j = findmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = qⱼ - (a - qᵢ)
        q[i] = a
    end
    K, V
end

function vmarsaglia!(K::Vector{Int}, V::Vector{T}, q::Vector{T}, p::Vector{T}) where {T<:AbstractFloat}
    (length(K) == length(V) == length(q) == length(p)) || throw(ArgumentError("all inputs must be of same size"))
    n = length(p)
    a = inv(n)
    @inbounds for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:n-1
        qᵢ, i = vfindmin(q)
        qⱼ, j = vfindmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = qⱼ - (a - qᵢ)
        q[i] = a
    end
    K, V
end

# faster, but not necessarily the method to use due to LoopVectorization and Base.Threads
# alas, it is ≈5x faster
function vmarsaglia_generate!(A::AbstractArray, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    n = length(K)
    u = rand(length(A))
    @turbo for i ∈ eachindex(A, u)
        j = trunc(Int, muladd(u[i], n, 1))
        A[i] = ifelse(u[i] < V[j], j, K[j])
    end
    A
end
function vmarsaglia_generate!(A::AbstractArray, u::AbstractArray{Float64}, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    n = length(K)
    rand!(u)
    @turbo for i ∈ eachindex(A, u)
        j = trunc(Int, muladd(u[i], n, 1))
        A[i] = ifelse(u[i] < V[j], j, K[j])
    end
    A
end

function vmarsaglia_generate(K::Vector{Int}, V::Vector{T}, dims::Vararg{Int, N}) where {T<:AbstractFloat} where {N}
    vmarsaglia_generate!(Array{Int}(undef, dims), K, V)
end

################
# convenience utils
@inline _marsaglia_init(T::Type{<:AbstractFloat}, n::Int) = Vector{Int}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n)
@inline _marsaglia_init(T::Type{<:AbstractFloat}) = _marsaglia_init(T, 0)
@inline _marsaglia_init() = _marsaglia_init(Float64)

@inline _genstorage_init(T::Type{<:AbstractFloat}, n::Int) = Vector{Int}(undef, n), Vector{T}(undef, n)

################
# experimental

mutable struct MarsagliaSquareHistogram{Ti<:Integer, Tv<:AbstractFloat}
    K::Vector{Ti}
    V::Vector{Tv}
    n::Int
end

# MarsagliaSquareHistogram{Ti, Tv}(K, V, n) where {Ti<:Integer, Tv<:AbstractFloat} =
#     MarsagliaSquareHistogram(convert(Vector{Ti}, K), convert(Vector{Tv}, V), n)

MarsagliaSquareHistogram(K, V) = MarsagliaSquareHistogram(K, V, length(K))

MarsagliaSquareHistogram((K, V), n) = MarsagliaSquareHistogram(K, V, n)
MarsagliaSquareHistogram(p) = MarsagliaSquareHistogram(marsaglia(p), length(p))

vmarsaglia_generate!(C, t::MarsagliaSquareHistogram) = ((; K, V, n) = t; vmarsaglia_generate!(C, K, V))
vmarsaglia_generate!(C, U, t::MarsagliaSquareHistogram) = ((; K, V, n) = t; vmarsaglia_generate!(C, U, K, V))
vmarsaglia_generate(t::MarsagliaSquareHistogram, dims::Vararg{Int, N}) where {N} =
    vmarsaglia_generate!(Array{Int}(undef, dims), t)


################
# Equal probability case admits an optimized form:
# U ~ Uniform(0,1)
# j = ⌊nU + 1⌋; return j
function vfill!(A::AbstractArray, v::Real)
    @turbo for i ∈ eachindex(A)
        A[i] = v
    end
    A
end

function vmarsaglia_generate!(A::AbstractArray, u::AbstractArray{Float64}, n::Int)
    n > 0 || throw(ArgumentError("n must be > 0"))
    n == 1 && return vfill!(A, 1)
    rand!(u)
    @turbo for i ∈ eachindex(A, u)
        A[i] = trunc(Int, muladd(u[i], n, 1))
    end
    A
end

vmarsaglia_generate!(A::AbstractArray, n::Int) = vmarsaglia_generate!(A, similar(A, Float64), n)

