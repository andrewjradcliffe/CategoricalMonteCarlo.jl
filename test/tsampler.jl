# Tests of tsampler functionality

@testset "tsampler, equal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                # # A simplification: an array of sparse vectors
                A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
            end
        end
    end
end

@testset "tsampler, unequal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                # # A simplification: an array of sparse vectors
                A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
            end
        end
    end
end
