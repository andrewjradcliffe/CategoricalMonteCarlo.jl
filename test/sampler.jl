# Tests of sampler functionality

@testset "sampler, equal probability mass" begin
    # # Specialized method for eltype(A)::Vector{Vector{Int}}
    A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
    B = sample(Int, A, 10, num_cat(A), (1,))
    @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
    @test all(minimum(B, dims=2) .≥ 0)
    # # A simplification: an array of sparse vectors
    A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    B = sample(Int, A, 10, num_cat(A), (1,))
    @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
    @test all(minimum(B, dims=2) .≥ 0)
end
