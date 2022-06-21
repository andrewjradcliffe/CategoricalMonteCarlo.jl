# Tests of Marsaglia's Robin Hood method

Random.seed!(1234)
@testset "Marsaglia: Robin Hood alias table" begin
    p = [2/15, 7/15, 6/15]
    K, V = marsaglia(p)
    @test K == [2,3,3]
    @test V ≈ [2/15, 9/15, 15/15]
    p = [.21, .18, .26, .17, .18]
    K, V = marsaglia(p)
    @test K == [1,3,1,3,3]
    @test V ≈ [0.2, 0.38, 0.5900000000000001, 0.7700000000000001, 0.98]
    #
    p = [2/15, 7/15, 6/15]
    K, V = marsaglia(p)
    #
    K′, V′ = Vector{Int}(), Vector{Float64}()
    ix, q = Vector{Int}(), Vector{Float64}()
    n = length(p)
    resize!(K′, n); resize!(V′, n); resize!(ix, n); resize!(q, n)
    marsaglia!(K′, V′, q, ix, p)
    @test K == K′
    @test V == V′
    #
    resize!(V′, 0)
    @test_throws ArgumentError marsaglia!(K′, V′, q, ix, p)
    @test_throws BoundsError marsaglia_generate(K′, V′)
end

@testset "Marsaglia: generate" begin
    p = [2/15, 7/15, 6/15]
    K, V = marsaglia(p)
    A = marsaglia_generate(K, V, 1,2,3);
    mn, mx = extrema(A)
    @test mn ≥ 1 && mx ≤ 3
    A = marsaglia_generate(K, V, 10^6);
    t = [count(==(i), A) for i = 1:length(p)]
    @test isapprox(t ./ 10^6, p, atol=1e-3)
    u = rand(10)
    @test_throws DimensionMismatch marsaglia_generate!(A, u, K, V)
    resize!(u, length(A))
    marsaglia_generate!(A, u, K, V)
    t = [count(==(i), A) for i = 1:length(p)]
    @test isapprox(t ./ 10^6, p, atol=1e-3)
    resize!(V, 0)
    @test_throws BoundsError marsaglia_generate(K, V)
    @test_throws ArgumentError marsaglia_generate!(A, K, V)
    resize!(V, 3)
    resize!(K, 0)
    @test_throws BoundsError marsaglia_generate(K, V)
    @test_throws ArgumentError marsaglia_generate!(A, K, V)
end

function countcategory(A::AbstractArray{T, N}) where {T<:Integer, N}
    mx = maximum(A)
    v = zeros(Int, mx)
    @inbounds @simd for i ∈ eachindex(A)
        v[A[i]] += 1
    end
    v
end

@testset "Marsaglia generate: numerical stability" begin
    n_samples = 10^8
    c = inv(n_samples)
    A = Vector{Int}(undef, n_samples)
    u = Vector{Float64}(undef, n_samples)
    for i = 1:15
        n = (1 << i)
        p = fill(1/n, n)
        K, V = marsaglia(p)
        marsaglia_generate!(A, u, K, V)
        t = countcategory(A);
        @test all(i -> isapprox(t[i] * c, p[i], atol=1e-3), 1:n)
        rand!(p)
        normalize1!(p)
        K, V = marsaglia(p)
        marsaglia_generate!(A, u, K, V)
        t = countcategory(A);
        @test all(i -> isapprox(t[i] * c, p[i], atol=1e-3), 1:n)
    end
end
