#
# Date created: 2022-06-21
# Author: aradclif
#
#
############################################################################################
# https://www.jstatsoft.org/article/view/v011i03
# Marsaglia's Robin Hood
# p ∈ ℝᴺ, ∑ᵢpᵢ = 1, a = 1/N
# K ∈ ℕᴺ, Kᵢ = i
# V ∈ ℝⁿ, Vᵢ = (i + 1) * a
# Generate: j = ⌊N*U⌋; if U < V[j], return j, else return K[j]
# it happens to not be faster to avoid the floor(Int, N * U), generating j=rand(1:N), and U=rand()
# separately.
# The objective is to minimize the number of times the `else return K[j]` occurs
# by creating V such that each V[j] is as close to 1 as possible -- an NP-hard problem,
# but Marsaglia's suggestion of the Robin Hood method (rob from rich to bring poor up to average)
# is a good solution which is 𝒪(NlogN).

function marsaglia(p::Vector{T}) where {T<:AbstractFloat}
    N = length(p)
    K = Vector{Int}(undef, N)
    V = Vector{promote_type(T, Float64)}(undef, N)
    ix = Vector{Int}(undef, N)
    q = similar(p)
    a = inv(N)
    # initialize
    for i ∈ eachindex(K, V, p, q)
        K[i] = i
        # V[i] = (i + 1) * a
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:N-1
        sortperm!(ix, q)
        i = ix[1]
        j = ix[N]
        K[i] = j
        V[i] = (i - 1) * a + q[i]
        # V[i] = i * a + q[i]
        q[j] = (q[j] + q[i]) - a # q[j] - (a - q[i])
        q[i] = a
    end
    K, V
end

p = [0.2, 0.3, 0.1, 0.4]

p = [.21, .18, .26, .17, .18]

p = [2/15, 7/15, 6/15]

function marsaglia_generate(K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    N = length(K)
    u = rand()
    j = floor(Int, muladd(u, N, 1))
    u < V[j] ? j : K[j]
end

using BenchmarkTools, Random

K, V = marsaglia(p)
@benchmark marsaglia_generate($K, $V)
@benchmark marsaglia_generate2($K, $V)
@benchmark marsaglia_generate3($K, $V)

p = rand(100);
normalize1!(p);
K, V = marsaglia(p);

function marsaglia_generate!(A::AbstractArray, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    N = length(K)
    @inbounds for i ∈ eachindex(A)
        u = rand()
        j = floor(Int, muladd(u, N, 1)) # muladd is faster than u * N + 1 by ≈5-6%
        A[i] = u < V[j] ? j : K[j]
    end
    A
end


function marsaglia_generate_simd!(A::AbstractArray, u::Vector{Float64}, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    length(K) == length(V) || throw(ArgumentError("K and V must be of same size"))
    N = length(K)
    @inbounds @simd for i ∈ eachindex(A, u)
        j = floor(Int, muladd(u[i], N, 1)) # muladd is faster than u * N + 1 by ≈5-6%
        A[i] = u[i] < V[j] ? j : K[j]
    end
    A
end
marsaglia_generate_simd!(A::AbstractArray, K::Vector{Int}, V::Vector{<:AbstractFloat}) = marsaglia_generate_simd!(A, rand(length(A)), K, V)



function marsaglia!(K::Vector{Int}, V::Vector{T}, q::Vector{T}, ix::Vector{Int}, p::Vector{T}) where {T<:AbstractFloat}
    (length(K) == length(V) == length(q) == length(ix) == length(p)) || throw(ArgumentError("all inputs must be of same size"))
    N = length(p)
    a = inv(N)
    @inbounds for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:N-1
        sortperm!(ix, q)
        i = ix[1]
        j = ix[N]
        K[i] = j
        V[i] = (i - 1) * a + q[i]
        q[j] = (q[j] + q[i]) - a
        q[i] = a
    end
    K, V
end
marsaglia2(p::Vector{T}) where {T<:AbstractFloat} =
    (N = length(p); marsaglia!(Vector{Int}(undef, N), Vector{promote_type(T, Float64)}(undef, N), similar(p), Vector{Int}(undef, N), p))


# faster, but not necessarily the method to use due to LoopVectorization and Base.Threads
# alas, it is ≈5x faster
function marsaglia_generate4!(A::AbstractArray, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    N = length(K)
    u = rand(length(A))
    @turbo for i ∈ eachindex(A, u)
        j = floor(Int, muladd(u[i], N, 1))
        A[i] = ifelse(u[i] < V[j], j, K[j])
    end
    A
end
function marsaglia_generate5!(A::AbstractArray, u::Vector{Float64}, K::Vector{Int}, V::Vector{T}) where {T<:AbstractFloat}
    N = length(K)
    rand!(u)
    @turbo for i ∈ eachindex(A, u)
        j = floor(Int, muladd(u[i], N, 1))
        A[i] = ifelse(u[i] < V[j], j, K[j])
    end
    A
end

n_samples = 1024
C = Vector{Int}(undef, n_samples);
@benchmark marsaglia_generate!($C, $K, $V)
@benchmark marsaglia_generate_simd!($C, $K, $V)
@benchmark marsaglia_generate2!($C, $K, $V)
@benchmark marsaglia_generate3!($C, $K, $V)
@benchmark marsaglia_generate4!($C, $K, $V)
@benchmark marsaglia_generate5!($C, $K, $V) # 3 with @inbounds
@benchmark marsaglia_generate6!($C, $K, $V) # 3 with @inbounds
@benchmark marsaglia_generate7!($C, $K, $V) # 3 with @inbounds
[[count(==(i), C) for i = 1:length(p)] ./ n_samples p]


# faster than nearly-divisionless? -- in fact, both are.
p = fill(1/10000, 10000);
K, V = marsaglia(p);
r = 1:10000
@benchmark rand!($C, $r)
x = rand(1024);
@benchmark rand!($x)
1024 / 2e-6

Σp = cumsum(p);
U = rand(length(C));
@benchmark categorical!($C, $U, $Σp)
