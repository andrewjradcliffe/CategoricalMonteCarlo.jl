# Tests of Marsaglia's Robin Hood method

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
end

@testset "Marsaglia: generate" begin
    Random.seed!(1234)
    p = [2/15, 7/15, 6/15]
    K, V = marsaglia(p)
    A = marsaglia_generate(K, V, 1,2,3);
    mn, mx = extrema(A)
    @test mn ≥ 1 && mx ≤ 3
    A = marsaglia_generate(K, V, 10^6);
    t = [count(==(i), A) for i = 1:length(p)]
    @test isapprox(t ./ 10^6, p, atol=1e-3)
end

