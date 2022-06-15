# Tests of tsampler functionality

@testset "tsampler, equal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (), (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
                @test all(==(3), sum(B, dims=1))
                @test all(sum(B, dims=2) .≤ n_sim .* [3; 3; 2; 2; 1; 1])
                # # A simplification: an array of sparse vectors
                A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
                @test all(==(3), sum(B, dims=(1,3)))
                @test all(sum(B, dims=(2,3)) .≤ n_sim .* [3; 3; 2; 2; 1; 1])
                # # The simplest case: a sparse vector
                A = [1,2,3,4,5,6]
                B = tsample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=1))
                A = [1,2,3,4]
                tsample!(B, A)
                @test all(==(2), sum(B, dims=1))
                @test all(maximum(B, dims=2) .≤ [2; 2; 2; 2; 1; 1])
                A = [1,2]
                tsample!(B, A)
                @test all(==(3), sum(B, dims=1))
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
            end
        end
    end
    @testset "eltypes" begin
        n_sim = 10
        A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128,
                 Float16, Float32, Float64, BigFloat, BigInt, Rational]
            B = @inferred tsample(T, A, n_sim)
            @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
            @test all(≥(0), minimum(B, dims=2))
            @test all(==(3), sum(B, dims=1))
            @test all(sum(B, dims=2) .≤ n_sim .* [3; 3; 2; 2; 1; 1])
            @test_throws MethodError tsample(Complex{T}, A, n_sim)
        end
        @test_throws InexactError tsample(Bool, A, n_sim)
    end
end

@testset "tsampler, unequal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (),  (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Vector{Vector{Int}}
                A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                @test all(==(3), sum(B, dims=1))
                @test all(sum(B, dims=2) .≤ n_sim .* [3; 3; 2; 2; 1; 1])
                # # A simplification: an array of sparse vectors
                A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
                B = tsample(Int, A, n_sim, num_cat(A), dims=region)
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(minimum(B, dims=2) .≥ 0)
                @test all(==(3), sum(B, dims=(1,3)))
                @test all(sum(B, dims=(2,3)) .≤ n_sim .* [3; 3; 2; 2; 1; 1])
                # # The simplest case: a sparse vector
                A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
                B = tsample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=1))
                A = ([1,2,3,4], [0.2, 0.3, 0.4, 0.1])
                tsample!(B, A)
                @test all(==(2), sum(B, dims=1))
                @test all(maximum(B, dims=2) .≤ [2; 2; 2; 2; 1; 1])
                A = ([1, 2], [0.3, 0.7])
                tsample!(B, A)
                @test all(==(3), sum(B, dims=1))
                @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
                @test all(≥(0), minimum(B, dims=2))
            end
        end
    end
    @testset "eltypes" begin
        n_sim = 10
        A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128,
                 Float16, Float32, Float64, BigFloat, BigInt, Rational]
            B = @inferred tsample(T, A, n_sim)
            @test all(maximum(B, dims=2) .≤ [3; 3; 2; 2; 1; 1])
            @test all(≥(0), minimum(B, dims=2))
            @test all(==(3), sum(B, dims=1))
            @test all(sum(B, dims=2) .≤ n_sim .* [3; 3; 2; 2; 1; 1])
            @test_throws MethodError tsample(Complex{T}, A, n_sim)
        end
        @test_throws InexactError tsample(Bool, A, n_sim)
    end
end

@testset "tsampler inferface throws" begin
    n_sim = 10
    A = [1,2,3,4,5,6]
    @test_throws MethodError tsample(Int, A, n_sim, dims=1:2)
    @test_throws MethodError tsample(Int, A, n_sim, dims=[1,2,3])
    @test_throws MethodError tsample(Int, A, n_sim, dims=[.1 .2; .3 .4])
    A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
    @test_throws MethodError tsample(Int, A, n_sim, dims=1:2)
    @test_throws MethodError tsample(Int, A, n_sim, dims=[1,2,3])
    @test_throws MethodError tsample(Int, A, n_sim, dims=[.1 .2; .3 .4])
end

@testset "tsampler, equal probability mass" begin
    n_sim = 100
    A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    D = fill(A, 10,5,5);
    # # Specialized method for eltype(A)::Vector{Vector{Int}}
    # Admittedly, not very meaningful test as
    Pr = 1/2 * 1/4 * 1/6
    lPr = length(D) * log(Pr) # necessary to even view as log probability
    lPr * log10(ℯ) # or on log10 scale
    B = tsample(Int, D, n_sim, num_cat(D), dims=(1,2,3))
    @test all(maximum(B, dims=2) .≤  length(D) .* [3; 3; 2; 2; 1; 1])
    @test all(minimum(B, dims=2) .≥ 0)
    @test all(==(length(A) * length(D)), sum(B, dims=1))
end
