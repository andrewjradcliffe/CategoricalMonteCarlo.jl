# Tests of vtsampler functionality

@testset "vtsampler, equal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (), (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Array{Vector{Int}}
                A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(≥(0), minimum(B, dims=1))
                @test all(==(3), sum(B, dims=2))
                @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
                # # A simplification: an array of sparse vectors
                A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(≥(0), minimum(B, dims=1))
                @test all(==(3), sum(B, dims=(2,3)))
                @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
                # # The simplest case: a sparse vector
                A = [1,2,3,4,5,6]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=2))
                A = [1,2,3,4]
                vtsample!(B, A)
                @test all(==(2), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
                A = [1,2]
                vtsample!(B, A)
                @test all(==(3), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(≥(0), minimum(B, dims=1))
            end
        end
    end
    @testset "eltypes" begin
        n_sim = 10
        # Types one would normally expect
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128,
                 Float16, Float32, Float64, BigFloat, BigInt, Rational]
            A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = [1,2,3,4,5,6]
            B = @inferred vtsample(T, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = [1,2,3,4]
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = [1,2]
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
        end
        # Composite numeric types
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
            A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            A = [1,2,3,4,5,6]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = [1,2,3,4]
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = [1,2]
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
        end
        # Real, AbstractFloat, Integer, Signed, Unsigned. work but should be avoided
        A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
        @test_throws InexactError vtsample(Bool, A, 1000)
        @test_throws MethodError vtsample(Union{Int16, Int32}, A, n_sim)
        B = Matrix{Union{Int16,Int32}}(undef, 6, 10)
        @test_throws MethodError vtsample!(B, A)
    end
end

@testset "sparse vtsampler, unequal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (),  (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Array{Vector{Int}}
                A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(minimum(B, dims=1) .≥ 0)
                @test all(==(3), sum(B, dims=2))
                @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
                # # A simplification: an array of sparse vectors
                A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(minimum(B, dims=1) .≥ 0)
                @test all(==(3), sum(B, dims=(2,3)))
                @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
                # # The simplest case: a sparse vector
                A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=2))
                A = ([1,2,3,4], [0.2, 0.3, 0.4, 0.1])
                vtsample!(B, A)
                @test all(==(2), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
                A = ([1, 2], [0.3, 0.7])
                vtsample!(B, A)
                @test all(==(3), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(≥(0), minimum(B, dims=1))
            end
        end
    end
    @testset "eltypes" begin
        n_sim = 10
        # Types one would normally expect
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128,
                 Float16, Float32, Float64, BigFloat, BigInt, Rational]
            A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
            B = @inferred vtsample(T, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = ([1,2,3,4], [0.2, 0.3, 0.4, 0.1])
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = ([1, 2], [0.3, 0.7])
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
        end
        # Composite numeric types
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
            A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = ([1,2,3,4], [0.2, 0.3, 0.4, 0.1])
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = ([1, 2], [0.3, 0.7])
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
        end
        # Real, AbstractFloat, Integer, Signed, Unsigned. work but should be avoided
        A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.5, 0.2, 0.2, 0.05,0.025, 0.025])]] # slight change to increase probability of Inexact throw
        @test_throws InexactError vtsample(Bool, A, 1000)
        @test_throws MethodError vtsample(Union{Int16, Int32}, A, n_sim)
        B = Matrix{Union{Int16,Int32}}(undef, 6, 10)
        @test_throws MethodError vtsample!(B, A)
    end
end

@testset "dense vtsampler, (un)equal probability mass" begin
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (),  (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Array{Vector{<:AbstractFloat}}
                A = [[[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(minimum(B, dims=1) .≥ 0)
                @test all(==(3), sum(B, dims=2))
                @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
                # # A simplification: an array of dense vectors
                A = [[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(minimum(B, dims=1) .≥ 0)
                @test all(==(3), sum(B, dims=(2,3)))
                @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
                # # The simplest case: a dense vector
                A = [0.1, 0.1, 0.1, 0.1,0.1, 0.5]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=2))
                A = [0.2, 0.3, 0.4, 0.1]
                vtsample!(B, A)
                @test all(==(2), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
                A = [0.3, 0.7]
                vtsample!(B, A)
                @test all(==(3), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(≥(0), minimum(B, dims=1))
            end
        end
    end
    @testset "eltypes" begin
        n_sim = 10
        # Types one would normally expect
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128,
                 Float16, Float32, Float64, BigFloat, BigInt, Rational]
            A = [[[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = [[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = [0.1, 0.1, 0.1, 0.1,0.1, 0.5]
            B = @inferred vtsample(T, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = [0.2, 0.3, 0.4, 0.1]
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = [0.3, 0.7]
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
        end
        # Composite numeric types
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
            A = [[[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            A = [[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            A = [0.1, 0.1, 0.1, 0.1,0.1, 0.5]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = [0.2, 0.3, 0.4, 0.1]
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = [0.3, 0.7]
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
        end
        # Real, AbstractFloat, Integer, Signed, Unsigned. work but should be avoided
        A = [[[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.5, 0.2, 0.2, 0.05,0.025, 0.025]]] # slight change to increase probability of Inexact throw
        @test_throws InexactError vtsample(Bool, A, 1000)
        @test_throws MethodError vtsample(Union{Int16, Int32}, A, n_sim)
        B = Matrix{Union{Int16,Int32}}(undef, 6, 10)
        @test_throws MethodError vtsample!(B, A)
    end
end

@testset "SparseVector vtsampler, unequal probability mass" begin
    sv1 = SparseVector(2, [1,2], [0.3, 0.7])
    sv2 = SparseVector(4, [1,2,3,4], [0.2, 0.3, 0.4, 0.1])
    sv3 = SparseVector(6, [1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
    for region ∈ [1, 2, 3, 4, 5, (1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (1,2,3), (1,2,4), (2,3,4), (1,2,3,4), :, (),  (1, 5), (2, 5), (1,2,5), (5,6,7)]
        for i = 1:15
            for j = -1:1
                n_sim = (1 << i) + j
                # # Specialized method for eltype(A)::Array{SparseVector{<:AbstractFloat}}
                A = [[sv1, sv2, sv3]]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(minimum(B, dims=1) .≥ 0)
                @test all(==(3), sum(B, dims=2))
                @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
                # # A simplification: an array of SparseVector
                A = [sv1, sv2, sv3]
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(minimum(B, dims=1) .≥ 0)
                @test all(==(3), sum(B, dims=(2,3)))
                @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
                # # The simplest case: a SparseVector
                A = sv3
                B = vtsample(Int, A, n_sim, dims=region)
                @test all(==(1), sum(B, dims=2))
                A = sv2
                vtsample!(B, A)
                @test all(==(2), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
                A = sv1
                vtsample!(B, A)
                @test all(==(3), sum(B, dims=2))
                @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
                @test all(≥(0), minimum(B, dims=1))
            end
        end
    end
    @testset "eltypes" begin
        n_sim = 10
        # Types one would normally expect
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128,
                 Float16, Float32, Float64, BigFloat, BigInt, Rational]
            A = [[sv1, sv2, sv3]]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = [sv1, sv2, sv3]
            B = @inferred vtsample(T, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
            A = sv3
            B = @inferred vtsample(T, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = sv2
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = sv1
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test_throws MethodError vtsample(Complex{T}, A, n_sim)
        end
        # Composite numeric types
        for T ∈ [Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
            A = [[sv1, sv2, sv3]]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=2))
            @test all(sum(B, dims=1) .≤ n_sim .* [3 3 2 2 1 1])
            A = [sv1, sv2, sv3]
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
            @test all(==(3), sum(B, dims=(2,3)))
            @test all(sum(B, dims=(1,3)) .≤ n_sim .* [3 3 2 2 1 1])
            A = sv3
            B = @inferred vtsample(Rational{T}, A, n_sim)
            @test all(==(1), sum(B, dims=2))
            A = sv2
            vtsample!(B, A)
            @test all(==(2), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [2 2 2 2 1 1])
            A = sv1
            vtsample!(B, A)
            @test all(==(3), sum(B, dims=2))
            @test all(maximum(B, dims=1) .≤ [3 3 2 2 1 1])
            @test all(≥(0), minimum(B, dims=1))
        end
        # Real, AbstractFloat, Integer, Signed, Unsigned. work but should be avoided
        A = [[sv1, sv2, SparseVector(6, [1,2,3,4,5,6], [0.5, 0.2, 0.2, 0.05,0.025, 0.025])]] # slight change to increase probability of Inexact throw
        @test_throws InexactError vtsample(Bool, A, 1000)
        @test_throws MethodError vtsample(Union{Int16, Int32}, A, n_sim)
        B = Matrix{Union{Int16,Int32}}(undef, 6, 10)
        @test_throws MethodError vtsample!(B, A)
    end
end
