#
# Date created: 2022-07-11
# Author: aradclif
#
#
############################################################################################
# Some benchmarks of normalization performance

#### normalize1
for i = 1:15
    for j = -1:1
        n = (1 << i) + j
        p = rand(n)
        println("normalize1!, n = ", n)
        @btime normalize1!($p)
        println("vnormalize1!, n = ", n)
        @btime vnormalize1!($p)
    end
end

w = [.1, .2, .3, .4, 0.0]
w = zeros(5)
p = similar(w)
algorithm3!(p, w, 0.0)

w = rand(2^6);
w[rand(1:2^6, 10)] .= 0;
p = similar(w);
u = 0.5
@benchmark algorithm3!($p, $w, $u)
@benchmark algorithm3_v2!($p, $w, $u)
@benchmark algorithm3_v3!($p, $w, $u)

#### Algorithm 3
# As one might expect, @turbo handles tails better than base julia
u = 0.5
for i = 1:15
    for j = -1:1
        n = (1 << i) + j
        w = rand(n)
        w[rand(1:n, n >> 1)] .= 0
        p = similar(w)
        println("algorithm3!, n = ", n)
        @btime algorithm3!($p, $w, $u)
        println("algorithm3_v2!, n = ", n)
        @btime algorithm3_v2!($p, $w, $u)
        println("algorithm3_v3!, n = ", n)
        @btime algorithm3_v3!($p, $w, $u)
        println("valgorithm3!, n = ", n)
        @btime valgorithm3!($p, $w, $u)
    end
end

function algorithm3_v3!(p::Vector{S}, w::Vector{T}, u::S) where {S<:AbstractFloat, T<:Real}
    s = zero(T)
    z = 0
    @inbounds @simd for i ∈ eachindex(p, w)
        w̃ = w[i]
        s += w̃
        p[i] = w̃
        z += w̃ == zero(T)
    end
    c = z == 0 ? inv(s) : (one(S) - u) / s
    u′ = z == length(p) ? inv(z) : u / z
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = ifelse(pᵢ == zero(S), u′, pᵢ * c)
    end
    p
end

function valgorithm3!(p::Vector{S}, w::Vector{T}, u::S) where {S<:AbstractFloat, T<:Real}
    s = zero(T)
    z = 0
    @turbo for i ∈ eachindex(p, w)
        w̃ = w[i]
        s += w̃
        p[i] = w̃
        z += w̃ == zero(T)
    end
    c = z == 0 ? inv(s) : (one(S) - u) / s
    u′ = z == length(p) ? inv(z) : u / z
    @turbo for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = ifelse(pᵢ == zero(S), u′, pᵢ * c)
    end
    p
end

#### Algorithm 2.2
algorithm2_2_quote(3)
algorithm2_2_normalize_quote(3)
algorithm2_2_normalize1!(p, Is, ws) = normalize1!(algorithm2_2!(p, Is, ws))

N = 2^10
n = 2^7
m = 3
Is = ntuple(_ -> rand(1:N, n), m);
ws = ntuple(_ -> rand(N), m);

p = zeros(n);
@benchmark algorithm2_2!($p, $Is, $ws)
@benchmark algorithm2_2_normalize1!($p, $Is, $ws)
@benchmark algorithm2_2_normalize!($p, $Is, $ws)
@benchmark valgorithm2_2_normalize!($p, $Is, $ws)
@timev algorithm2_2_normalize!(p, (Is[1],), (ws[1],))

str = """
@inbounds @simd ivdep for j = eachindex(I_1, I_2, I_3)
    w′[j] = w_1[I_1[j]] * w_2[I_2[j]] * w_3[I_3[j]]
end
"""
e = Meta.parse(str)
Meta.show_sexpr(e)
eq = :(@inbounds @simd ivdep for j = eachindex(I_1, I_2, I_3)
           w′[j] = w_1[I_1[j]] * w_2[I_2[j]] * w_3[I_3[j]]
       end)
Meta.show_sexpr(eq)
