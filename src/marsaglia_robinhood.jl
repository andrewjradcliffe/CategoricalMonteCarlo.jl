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

# function marsaglia(p::Vector{T}) where {T<:AbstractFloat}
#     N = length(p)
#     K = Vector{Int}(undef, N)
#     V = Vector{promote_type(T, Float64)}(undef, N)
#     ix = Vector{Int}(undef, N)
#     q = similar(p)
#     a = inv(N)
#     # initialize
#     for i ∈ eachindex(K, V, p, q)
#         K[i] = i
#         V[i] = i * a
#         q[i] = p[i]
#     end
#     for _ = 1:N-1
#         sortperm!(ix, q)
#         i = ix[1]
#         j = ix[N]
#         K[i] = j
#         V[i] = (i - 1) * a + q[i]
#         q[j] = (q[j] + q[i]) - a # q[j] - (a - q[i])
#         q[i] = a
#     end
#     K, V
# end

function marsaglia(p::Vector{T}) where {T<:AbstractFloat}
    N = length(p)
    K = Vector{Int}(undef, N)
    V = Vector{promote_type(T, Float64)}(undef, N)
    q = similar(p)
    a = inv(N)
    # initialize
    for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:N-1
        qᵢ, i = findmin(q)
        qⱼ, j = findmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = (qⱼ + qᵢ) - a
        q[i] = a
    end
    K, V
end

function vmarsaglia(p::Vector{T}) where {T<:AbstractFloat}
    N = length(p)
    K = Vector{Int}(undef, N)
    V = Vector{promote_type(T, Float64)}(undef, N)
    a = inv(N)
    q = similar(p)
    # initialize
    for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:N-1
        qᵢ, i = vfindmin(q)
        qⱼ, j = vfindmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = (qⱼ + qᵢ) - a
        q[i] = a
    end
    K, V
end

# p = [0.2, 0.3, 0.1, 0.4]

# p = [.21, .18, .26, .17, .18]

# p = [2/15, 7/15, 6/15]

function marsaglia_generate(K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    N = length(K)
    u = rand()
    j = floor(Int, muladd(u, N, 1))
    u < V[j] ? j : K[j]
end

# using BenchmarkTools, Random

# K, V = marsaglia(p)
# @benchmark marsaglia_generate($K, $V)
# @benchmark marsaglia_generate2($K, $V)
# @benchmark marsaglia_generate3($K, $V)

# p = rand(100);
# normalize1!(p);
# K, V = marsaglia(p);

function marsaglia_generate!(A::AbstractArray, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    N = length(K)
    @inbounds for i ∈ eachindex(A) # safe to also use @fastmath, @simd
        u = rand()
        j = floor(Int, muladd(u, N, 1)) # muladd is faster than u * N + 1 by ≈5-6%
        A[i] = u < V[j] ? j : K[j]
    end
    A
end
function marsaglia_generate!(A::AbstractArray, u::AbstractArray{Float64}, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    N = length(K)
    rand!(u)
    @inbounds for i ∈ eachindex(A, u) # safe to also use @fastmath, @simd
        j = floor(Int, muladd(u[i], N, 1)) # muladd is faster than u * N + 1 by ≈5-6%
        A[i] = u[i] < V[j] ? j : K[j]
    end
    A
end

function marsaglia_generate(K::Vector{Int}, V::Vector{T}, dims::Vararg{Int, N}) where {T<:AbstractFloat} where {N}
    marsaglia_generate!(Array{Int}(undef, dims), K, V)
end

# function marsaglia!(K::Vector{Int}, V::Vector{T}, q::Vector{T}, ix::Vector{Int}, p::Vector{T}) where {T<:AbstractFloat}
#     (length(K) == length(V) == length(q) == length(ix) == length(p)) || throw(ArgumentError("all inputs must be of same size"))
#     N = length(p)
#     a = inv(N)
#     @inbounds for i ∈ eachindex(K, V, p, q)
#         K[i] = i
#         V[i] = i * a
#         q[i] = p[i]
#     end
#     for _ = 1:N-1
#         sortperm!(ix, q)
#         i = ix[1]
#         j = ix[N]
#         K[i] = j
#         V[i] = (i - 1) * a + q[i]
#         q[j] = (q[j] + q[i]) - a
#         q[i] = a
#     end
#     K, V
# end
# marsaglia2(p::Vector{T}) where {T<:AbstractFloat} =
#     (N = length(p); marsaglia!(Vector{Int}(undef, N), Vector{promote_type(T, Float64)}(undef, N), similar(p), Vector{Int}(undef, N), p))

function marsaglia!(K::Vector{Int}, V::Vector{T}, q::Vector{T}, p::Vector{T}) where {T<:AbstractFloat}
    (length(K) == length(V) == length(q) == length(p)) || throw(ArgumentError("all inputs must be of same size"))
    N = length(p)
    a = inv(N)
    @inbounds for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:N-1
        qᵢ, i = findmin(q)
        qⱼ, j = findmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = (qⱼ + qᵢ) - a
        q[i] = a
    end
    K, V
end

function vmarsaglia!(K::Vector{Int}, V::Vector{T}, q::Vector{T}, p::Vector{T}) where {T<:AbstractFloat}
    (length(K) == length(V) == length(q) == length(p)) || throw(ArgumentError("all inputs must be of same size"))
    N = length(p)
    a = inv(N)
    @inbounds for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:N-1
        qᵢ, i = vfindmin(q)
        qⱼ, j = vfindmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = (qⱼ + qᵢ) - a
        q[i] = a
    end
    K, V
end

# faster, but not necessarily the method to use due to LoopVectorization and Base.Threads
# alas, it is ≈5x faster
function vmarsaglia_generate!(A::AbstractArray, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    N = length(K)
    u = rand(length(A))
    @turbo for i ∈ eachindex(A, u)
        j = floor(Int, muladd(u[i], N, 1))
        A[i] = ifelse(u[i] < V[j], j, K[j])
    end
    A
end
function vmarsaglia_generate!(A::AbstractArray, u::AbstractArray{Float64}, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    N = length(K)
    rand!(u)
    @turbo for i ∈ eachindex(A, u)
        j = floor(Int, muladd(u[i], N, 1))
        A[i] = ifelse(u[i] < V[j], j, K[j])
    end
    A
end

# n_samples = 1024
# C = Vector{Int}(undef, n_samples);
# @benchmark marsaglia_generate!($C, $K, $V)
# @benchmark marsaglia_generate_simd!($C, $K, $V)
# @benchmark marsaglia_generate2!($C, $K, $V)
# @benchmark marsaglia_generate3!($C, $K, $V)
# @benchmark marsaglia_generate4!($C, $K, $V)
# @benchmark marsaglia_generate5!($C, $K, $V) # 3 with @inbounds
# @benchmark marsaglia_generate6!($C, $K, $V) # 3 with @inbounds
# @benchmark marsaglia_generate7!($C, $K, $V) # 3 with @inbounds
# [[count(==(i), C) for i = 1:length(p)] ./ n_samples p]


# # faster than nearly-divisionless? -- in fact, both are.
# p = fill(1/10000, 10000);
# K, V = marsaglia(p);
# r = 1:10000
# @benchmark rand!($C, $r)
# x = rand(1024);
# @benchmark rand!($x)
# 1024 / 2e-6

# Σp = cumsum(p);
# U = rand(length(C));
# @benchmark categorical!($C, $U, $Σp)

################
# convenience utils
@inline _marsaglia_init(T::Type{<:AbstractFloat}, n::Int) = Vector{Int}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n)
@inline _marsaglia_init(T::Type{<:AbstractFloat}) = _marsaglia_init(T, 0)
@inline _marsaglia_init() = _marsaglia_init(Float64)

@inline _genstorage_init(T::Type{<:AbstractFloat}, n::Int) = Vector{Int}(undef, n), Vector{T}(undef, n)
