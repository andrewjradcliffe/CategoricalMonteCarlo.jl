#
# Date created: 2022-06-22
# Author: aradclif
#
#
############################################################################################
# Performance tests and variants of Marsaglia alias table generation (square histogram)

function marsaglia4(p::Vector{T}) where {T<:AbstractFloat}
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

function vmarsaglia4(p::Vector{T}) where {T<:AbstractFloat}
    N = length(p)
    K = Vector{Int}(undef, N)
    V = Vector{promote_type(T, Float64)}(undef, N)
    a = inv(N)
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
function vmarsaglia5(p::Vector{T}) where {T<:AbstractFloat}
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
        ((qᵢ, i), (qⱼ, j)) = vfindextrema(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = (qⱼ + qᵢ) - a
        q[i] = a
    end
    K, V
end
function vmarsaglia6(p::Vector{T}) where {T<:AbstractFloat}
    N = length(p)
    K = Vector{Int}(undef, N)
    V = Vector{promote_type(T, Float64)}(undef, N)
    # ix = Vector{Int}(undef, N)
    q = similar(p)
    a = inv(N)
    # initialize
    @turbo for i ∈ eachindex(K, V, p, q)
        K[i] = i
        V[i] = i * a
        q[i] = p[i]
    end
    for _ = 1:N-1
        ((qᵢ, i), (qⱼ, j)) = vfindextrema(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = (qⱼ + qᵢ) - a
        q[i] = a
    end
    K, V
end
for i = 1:10
    n = (1 << i)
    p = normalize1!(rand(n));
    println("n = $n, marsaglia")
    @btime marsaglia($p)
    println("n = $n, marsaglia4")
    @btime marsaglia4($p)
    println("n = $n, vmarsaglia4")
    @btime vmarsaglia4($p)
    println("n = $n, vmarsaglia5")
    @btime vmarsaglia5($p)
    println("n = $n, vmarsaglia6")
    @btime vmarsaglia6($p)
end
@benchmark marsaglia($p)
@benchmark marsaglia4($p)
@benchmark vmarsaglia4($p)
@benchmark vmarsaglia5($p)
@benchmark vmarsaglia6($p)
marsaglia(p) == marsaglia4(p)
vmarsaglia4(p) == vmarsaglia5(p)
K1, V1 = marsaglia(p)
K2, V2 = marsaglia4(p)
K2_4, V2_4 = marsaglia4_2(p)
K2_5, V2_5 = marsaglia5(p)
K3, V3 = vmarsaglia4(p)
K4, V4 = vmarsaglia5(p)
K5, V5 = vmarsaglia6(p)
K1 == K2 == K3 == K4 == K5
V1 == V2 == V3 == V4 == V5
K3 == K4 == K5
V3 == V4 == V5
ii23 = findall(K2 .!= K3)
ii34 = findall(K3 .!= K4)
ii35 = findall(K3 .!= K5)
ii45 = findall(K4 .!= K5)

K, V, q = marsaglia4_2(p);
K0, V0, q0 = marsaglia(p);
K2, V2, q2 = marsaglia4(p);
K4, V4, q4 = vmarsaglia4(p);
ii = findall(K2 .!= K4)
[V0[ii] V[ii] V2[ii] V4[ii]]

