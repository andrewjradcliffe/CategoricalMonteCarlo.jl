# Tests of sampler functionality

@testset "sampler, equal probability mass" begin
    A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
    B = sample(Int, A, 10, num_cat(A), (1,))
    @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
    @test all(minimum(B, dims=2) .≥ 0)
end
