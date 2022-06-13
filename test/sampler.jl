# Tests of sampler functionality

@testset "sampler, equal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                # # A simplification: an array of sparse vectors
                A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
            end
        end
    end
end

@testset "sampler, unequal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                # # A simplification: an array of sparse vectors
                A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
                B = sample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
            end
        end
    end
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

@testset "_check_reducedims" begin
    A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    D = fill(A, 10,5,5);
    @test_throws DimensionMismatch _check_reducedims(Matrix{Int}(undef, 0,2), D)
    @test_throws DimensionMismatch _check_reducedims(Matrix{Int}(undef, 2,2), D)
    @test_throws DimensionMismatch _check_reducedims(Matrix{Int}(undef, 3,2), D)
    @test_throws DimensionMismatch _check_reducedims(Matrix{Int}(undef, 4,2), D)
    @test_throws DimensionMismatch _check_reducedims(Matrix{Int}(undef, 5,2), D)
    @test _check_reducedims(Matrix{Int}(undef, 6,2), D)
    @test _check_reducedims(Matrix{Int}(undef, 60,2), D)
    @test _check_reducedims(Matrix{Int}(undef, 6,1), D)
    @test _check_reducedims(Matrix{Int}(undef, 60,1), D)
    @test_throws DimensionMismatch _check_reducedims(Array{Int}(undef, 6,1,2), D)
    @test _check_reducedims(Array{Int}(undef, 6,1,1), D)
    @test_throws DimensionMismatch _check_reducedims(Array{Int}(undef, 6,1,10,2), D)
    @test _check_reducedims(Array{Int}(undef, 6,1,10), D)
    @test_throws DimensionMismatch _check_reducedims(Array{Int}(undef, 6,1,10,2,5), D)
    @test _check_reducedims(Array{Int}(undef, 6,1,10,1,5), D)
    @test _check_reducedims(Array{Int}(undef, 6,1,10,5,5), D)
end

@testset "num_cat" begin
    A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    @test num_cat(A) == 6
    A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
    @test num_cat(A) == 6
    A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
    @test num_cat(A) == 6
    A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
    @test num_cat(A) == 6
    # empty cases
    @test num_cat(Vector{Vector{Int}}()) == 0
    @test num_cat(Vector{Vector{Vector{Int}}}()) == 0
    @test num_cat(Vector{Tuple{Vector{Int}, Vector{Float64}}}()) == 0
    @test num_cat(Vector{Vector{Tuple{Vector{Int}, Vector{Float64}}}}()) == 0
    # partially empty
    A = [[1, 2], Int[], [1, 2, 3, 4, 5, 6]]
    @test num_cat(A) == 6
    A = [[[1, 2], [1, 2, 3, 4], Int[]]]
    @test num_cat(A) == 4
    A = [([1, 2], [0.3, 0.7]), (Int[], Float64[]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
    @test num_cat(A) == 6
    A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), (Int[], Float64[])]]
    @test num_cat(A) == 4
end
