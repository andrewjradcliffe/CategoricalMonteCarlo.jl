# Tests of sampler functionality

@testset "sampler, equal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (), (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
                @test all(==(3), sum(B, dims=1))
                # # A simplification: an array of sparse vectors
                A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
                @test all(==(3), sum(B, dims=(1,3)))
                # # The simplest case: a sparse vector
                A = [1,2,3,4,5,6]
                B = sample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=1))
                A = [1,2,3,4]
                sample!(B, A)
                @test all(==(2), sum(B, dims=1))
                A = [1,2]
                sample!(B, A)
                @test all(==(3), sum(B, dims=1))
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
            end
        end
    end
end

@testset "sampler, unequal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (),  (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                @test all(==(3), sum(B, dims=1))
                # # A simplification: an array of sparse vectors
                A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                @test all(==(3), sum(B, dims=(1,3)))
                # # The simplest case: a sparse vector
                A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
                B = sample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=1))
                A = ([1,2,3,4], [0.2, 0.3, 0.4, 0.1])
                sample!(B, A)
                @test all(==(2), sum(B, dims=1))
                A = ([1, 2], [0.3, 0.7])
                sample!(B, A)
                @test all(==(3), sum(B, dims=1))
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
            end
        end
    end
end

@testset "sampler inferface throws" begin
    n_sim = 10
    A = [1,2,3,4,5,6]
    @test_throws MethodError sample(Int, A, n_sim, dims=1:2)
    @test_throws MethodError sample(Int, A, n_sim, dims=[1,2,3])
    @test_throws MethodError sample(Int, A, n_sim, dims=[.1 .2; .3 .4])
    A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
    @test_throws MethodError sample(Int, A, n_sim, dims=1:2)
    @test_throws MethodError sample(Int, A, n_sim, dims=[1,2,3])
    @test_throws MethodError sample(Int, A, n_sim, dims=[.1 .2; .3 .4])
end

@testset "sampler, equal probability mass" begin
    n_sim = 100
    A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    D = fill(A, 10,5,5);
    # # Specialized method for eltype(A)::Vector{Vector{Int}}
    # Admittedly, not very meaningful test as
    Pr = 1/2 * 1/4 * 1/6
    lPr = length(D) * log(Pr) # necessary to even view as log probability
    lPr * log10(ℯ) # or on log10 scale
    B = sample(Int, D, n_sim, num_cat(D), dims=(1,2,3))
    @test all(maximum(B, dims=2) .≤  length(D) .* [3; 3; 2; 2; 1; 1])
    @test all(minimum(B, dims=2) .≥ 0)
end
