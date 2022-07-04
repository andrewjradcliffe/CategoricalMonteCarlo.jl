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
        q[j] = qⱼ - (a - qᵢ)
        q[i] = a
    end
    K, V
end

function vmarsaglia4(p::Vector{T}) where {T<:AbstractFloat}
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
        qᵢ, i = vfindmin(q)
        qⱼ, j = vfindmax(q)
        K[i] = j
        V[i] = (i - 1) * a + qᵢ
        q[j] = qⱼ - (a - qᵢ)
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
        q[j] = qⱼ - (a - qᵢ)
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
        q[j] = qⱼ - (a - qᵢ)
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

#### Generation benchmarks
using BenchmarkTools, Random

K, V = marsaglia(p)
@benchmark marsaglia_generate($K, $V)
@benchmark marsaglia_generate2($K, $V)
@benchmark marsaglia_generate3($K, $V)

p = rand(100);
normalize1!(p);
K, V = marsaglia(p);

n_samples = 1024
C = Vector{Int}(undef, n_samples);
@benchmark marsaglia_generate!($C, $K, $V)
@benchmark vmarsaglia_generate!($C, $K, $V)

U = similar(C, Float64);
@benchmark marsaglia_generate!($C, $U, $K, $V)
@benchmark vmarsaglia_generate!($C, $U, $K, $V)

[[count(==(i), C) for i = 1:length(p)] ./ n_samples p]

# Equal probability comparison
p = fill(1/100, 100);
K, V = marsaglia(p);
@benchmark vmarsaglia_generate!($C, $U, $K, $V)
@benchmark vmarsaglia_equiprobable!($C, $U, 100)
ur = 1:100
@benchmark rand!($C, $ur)


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



#### Numerical stability tests
n = 10^3
p = normalize1!(rand(n));
p_b = big.(p);

K1, V1 = marsaglia(p);
K2, V2 = marsaglia2(p);
vK1, vV1 = vmarsaglia(p);
bK1, bV1 = marsaglia(p_b);
bK2, bV2 = marsaglia2(p_b);
K1 == K2
V1 == V2
K1 == vK1
V1 == vV1
bV1 == V1
bV1 == bV2
bV1 == V2
bV2 == V2
count(V1 .== V2)
count(bV1 .== V1)
count(bV2 .== V2)
extrema(V1 .- V2)
extrema(bV1 .- V1)
extrema(bV2 .- V2)

sum(abs, bV1 .- V1)
sum(abs, bV2 .- V2)

# a case which is unstable
p₁ = 0.999
# n = 10^4
for i = 1:10
    # n = (1 << i)
    n = 10^i
    p = [p₁; fill((1.0 - p₁) / n, n)];
    K1, V1 = marsaglia(p);
    K2, V2 = marsaglia2(p);
    @test K1 == K2
    @test V1 == V2
end
