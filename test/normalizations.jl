# Tests of normalization methods

@testset "misc type inference functions" begin
    @test _typeofinv(1) === Float64
    @test _typeofinv(Int) === Float64
    @test _typeofinv(Rational{Int64}) === Rational{Int64}
    #
    ws = ([1,2,3], fill(1/6, 6), fill(1//10, 9))
    @test _typeofprod(ws) === Float64
    ws2 = (ws[1], fill(1//6, 6), ws[3])
    @test _typeofprod(ws2) === Rational{Int64}
    for T ∈ (Int8, Int16, Int32)
        ws3 = (Int8[1,2,3], T[4,5,6], Int8[7,8,9])
        @test _typeofprod(ws3) === Int64
    end
end

@testset "_u, _checku01" begin
    @test _u(1.0) === 0.5
    @test _u(1//1) === 1//2
    @test _u(0) === 0.0
    @test _u(Inf) === 1.0
    for T ∈ (Int64, UInt, Float64, BigInt, BigFloat, Rational{Int})
        @test_nowarn _check_u01(zero(T))
        @test_nowarn _check_u01(one(T))
        @test_throws DomainError _check_u01(T(2))
    end
    @test_nowarn _check_u01(0.5)
    @test_nowarn _check_u01(1//2)
    @test_nowarn _check_u01(big(0.5))
end

@testset "algorithm 2.1" begin
    w = [7.3, 10.2, 5.1, 2.7, 2.89]
    p = [1.0]
    for i ∈ eachindex(w)
        @test algorithm2_1([i], w) ≈ p
    end
    for I ∈ ([1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5],
             [1, 2, 3], [1, 2, 4], [1, 2, 5], [2, 3, 4], [2, 3, 5], [3, 4, 5],
             [1, 2, 3, 4], [1, 2, 3, 5], [2, 3, 4, 5],
             [1, 2, 3, 4, 5])
        @test algorithm2_1(I, w) ≈ w[I] ./ sum(w[I])
    end
    @test algorithm2_1([1, 1, 1], w) ≈ fill(1/3, 3)
    # non-Float64 Type handling
    I1 = [1]
    I2 = [1, 2]
    for T ∈ (Float16, Float32, Rational{Int16}, Rational{Int32}, Rational{Int64}, Rational{Int128})
        𝑤 = T.(w)
        @test @inferred algorithm2_1(I1, 𝑤) ≈ T[1]
        @test @inferred algorithm2_1(I2, 𝑤) ≈ 𝑤[I2] ./ sum(𝑤[I2])
    end

    # Aberrant behavior
    @testset "weight < 0" begin
        w = [-5, 4, 3, 2, 1];
        I = [1, 5, 2];
        @test algorithm2_1(I, w) == [-Inf, Inf, Inf]
        I = [2, 3]
        @test algorithm2_1(I, w) == [4/7, 3/7]
        I = [1, 2, 2]
        @test algorithm2_1(I, w) == [-prevfloat(5/3), 4/3, 4/3]
    end
    # zeros behavior
    w = zeros(5)
    I = [1, 2, 3]
    @test all(algorithm2_1(I, w) .=== [-NaN, -NaN, -NaN])
    w[3] = 5
    @test algorithm2_1(I, w) == [0.0, 0.0, 1.0]

    @testset "NaN handling (lack thereof)" begin
        # things which generate NaNs by propagation
        w = [2.0, 10.0, 5.0, 1.0, NaN]
        I = [1, 3, 5]
        @test all(algorithm2_1(I, w) .=== [NaN, NaN, NaN])
        I = [1, 5, 5]
        @test all(algorithm2_1(I, w) .=== [NaN, NaN, NaN])
        I = [5, 5, 5]
        @test all(algorithm2_1(I, w) .=== [NaN, NaN, NaN])

        # propagating NaNs with signbit set
        w = [2.0, 10.0, 5.0, 1.0, -NaN]
        I = [1, 3, 5]
        @test all(algorithm2_1(I, w) .=== [-NaN, -NaN, -NaN])
        I = [1, 5, 5]
        @test all(algorithm2_1(I, w) .=== [-NaN, -NaN, -NaN])
        I = [5, 5, 5]
        @test all(algorithm2_1(I, w) .=== [-NaN, -NaN, -NaN])
    end

    @testset "±Inf handling (lack thereof)" begin
        w = [2.0, 10.0, 5.0, 1.0, Inf]
        I = [1, 3, 5]
        @test all(algorithm2_1(I, w) .=== [0.0, 0.0, -NaN])
        w = [2.0, 10.0, 5.0, 1.0, prevfloat(Inf)]
        I = [1, 3, 5]
        p = algorithm2_1(I, w)
        @test all(!isnan, p)
        @test all(!isinf, p)
        # integer overflow
        w = [5, 4, 3, 1, typemax(Int) - 1]
        I = [1, 3, 5]
        @test algorithm2_1(I, w) == [-5.421010862427522e-19, -3.2526065174565133e-19, -1.0]
        I = [4, 5] # on the cusp of overflow
        @test algorithm2_1(I, w) == [1.0842021724855044e-19, 1.0]
    end

    @testset "sweep precision (SIMD rounding)" begin
        rng = Xoshiro(1234)
        w = rand(rng, 256)
        p = Vector{Float64}(undef, 0)
        for m = -10:10
            w .*= 10.0^m
            for i = 1:8
                for j = -1:1
                    n = (1 << i) + j
                    I = rand(1:256, n)
                    resize!(p, n)
                    algorithm2_1!(p, I, w)
                    @test all(!iszero, p)
                    @test sum(p) ≈ 1.0
                end
            end
        end
    end

    @testset "algorithm 2.1, type handling" begin
        w1 = [5/9, 1/9, 3/9, 0, 0]
        w2 = [5//9, 1//9, 3//9, 0, 0]
        w3 = [5, 1, 3, 0, 0]
        p1 = [prevfloat(5/6), prevfloat(1/6), 0.0]
        p2 = [5//6, 1//6, 0]
        p3 = [prevfloat(5/6), 1/6, 0.0]
        I = [1, 2, 5]
        @test @inferred algorithm2_1(I, w) == p1
        @test @inferred algorithm2_1(I, w2) == p2
        @test @inferred algorithm2_1(I, w3) == p3
        p = similar(p1)
        @test @inferred algorithm2_1!(p, I, w1) ≈ p3
        @test @inferred algorithm2_1!(p, I, w2) ≈ p3
        @test @inferred algorithm2_1!(p, I, w3) ≈ p3
        #
        for T ∈ (Float16, Float32, Float64, BigFloat, Int8, Int16, Int32, Int64, BigInt, Rational{Int8}, Rational{Int16}, Rational{Int32}, Rational{Int64}, Rational{Int128}, Rational{BigInt})
            𝑤 = T.(w3)
            p′ = @inferred algorithm2_1(I, 𝑤)
            @test p′ ≈ p3
        end
    end
end

# _alg2_2(Is, ws) = (p = (.*)(getindex.(ws, Is)...); p .* inv(sum(p)))

@testset "algorithm 2.2" begin
    ws = ([1,2,3], fill(1/6, 6), fill(1//10, 9))
    ws2 = (ws[1], fill(1//6, 6), ws[3])
    Is = ([1,2,3], [4,5,6], [7,8,9])
    p = [1.0]
    p2 = [1//1]
    for i ∈ 1:3
        I = [i]
        @test @inferred algorithm2_2((I, I, I), ws) == p
        @test @inferred algorithm2_2((I, I, I), ws2) == p2
    end
    p′ = [1/6, 1/3, 1/2]
    p2′ = [1//6, 1//3, 1//2]
    @test @inferred algorithm2_2(Is, ws) == p′
    @test @inferred algorithm2_2(Is, ws2) == p2′
    # Float products
    I = [1,2,3]
    w = fill(1/2,6)
    for n = 1:10
        Is = ntuple(_ -> I, n)
        ws = ntuple(_ -> w, n)
        p = @inferred algorithm2_2(Is, ws)
        @test all(p .≈ 1/3)
    end
    # Rational products
    w2 = fill(1//2,6)
    for n = 1:10
        Is = ntuple(_ -> I, n)
        ws = ntuple(_ -> w2, n)
        p = @inferred algorithm2_2(Is, ws)
        @test all(p .== 1//3)
    end
    # Mixed products
    w3 = fill(1,6)
    for n = 2:2:12
        Is = ntuple(_ -> I, n)
        ws = tuple(ntuple(_ -> w, n >> 1)..., ntuple(_ -> w2, n >> 1)...)
        p = @inferred algorithm2_2(Is, ws)
        @test all(p .≈ 1/3)
        ws = tuple(ntuple(_ -> w, n >> 1)..., ntuple(_ -> w3, n >> 1)...)
        p = @inferred algorithm2_2(Is, ws)
        @test all(p .≈ 1/3)
        Base.setindex(ws, w2, lastindex(ws))
        p = @inferred algorithm2_2(Is, ws)
        @test all(p .≈ 1/3)
    end
    # Aberrant behavior
    @testset "weight < 0" begin
        w = [-5, 4, 3, 2, 1];
        I = [1, 5, 2];
        Is = (I, [1,2,3], [1,2,3])
        ws = (w, fill(1/6, 6), fill(1//10, 9))
        @test algorithm2_2(Is, ws) == [-Inf, Inf, Inf]
        Is = Base.setindex(Is, [2,3,4], 1)
        @test algorithm2_2(Is, ws) == nextfloat.([4/9, 3/9, 2/9])
        Is = Base.setindex(Is, [1,2,2], 1)
        @test algorithm2_2(Is, ws) == [-prevfloat(5/3), 4/3, 4/3]
    end
    # zeros behavior
    w = zeros(5)
    ws = (w, w, w)
    I = [1, 2, 3]
    Is = (I, I, I)
    @test all(algorithm2_2(Is, ws) .=== [-NaN, -NaN, -NaN])
    w[3] = 5
    @test algorithm2_2(Is, ws) == [0.0, 0.0, 1.0]

    @testset "NaN handling (lack thereof)" begin
        # things which generate NaNs by propagation
        w = [2.0, 10.0, 5.0, 1.0, NaN]
        ws = (w, w, w)
        I = [1, 3, 5]
        Is = (I, I, I)
        @test all(algorithm2_2(Is, ws) .=== [NaN, NaN, NaN])
        I = [1, 5, 5]
        Is = (I, I, I)
        @test all(algorithm2_2(Is, ws) .=== [NaN, NaN, NaN])
        I = [5, 5, 5]
        Is = (I, I, I)
        @test all(algorithm2_2(Is, ws) .=== [NaN, NaN, NaN])

        # propagating NaNs with signbit set
        w = [2.0, 10.0, 5.0, 1.0, -NaN]
        ws = (w, w, w)
        I = [1, 3, 5]
        Is = (I, I, I)
        @test all(algorithm2_2(Is, ws) .=== [-NaN, -NaN, -NaN])
        I = [1, 5, 5]
        Is = (I, I, I)
        @test all(algorithm2_2(Is, ws) .=== [-NaN, -NaN, -NaN])
        I = [5, 5, 5]
        Is = (I, I, I)
        @test all(algorithm2_2(Is, ws) .=== [-NaN, -NaN, -NaN])
    end

    @testset "±Inf handling (lack thereof)" begin
        I = [1, 3, 5]
        Is = (I, I, I)
        w = [2.0, 10.0, 5.0, 1.0, Inf]
        ws = (w, w, w)
        @test all(algorithm2_2(Is, ws) .=== [0.0, 0.0, -NaN])
        w = [2.0, 10.0, 5.0, 1.0, prevfloat(Inf)]
        ws = (w, w, w)
        p = algorithm2_2(Is, ws)
        @test all(p .=== [0.0, 0.0, -NaN])
        @test all(!isinf, p)
        # integer overflow
        w = [5, 4, 3, 1, typemax(Int) - 1]
        ws = (w, w, w)
        @test algorithm2_2(Is, ws) == [0.8680555555555555, 0.1875, -0.05555555555555555]
    end

    @testset "sweep precision (SIMD rounding)" begin
        rng = Xoshiro(0xd123456789abcdef)
        w = rand(rng, 256)
        p = Vector{Float64}(undef, 0)
        for M = 2:5
            for m = -10:10
                w .*= 10.0^m
                ws = ntuple(_ -> w, M)
                for i = 1:8
                    for j = -1:1
                        n = (1 << i) + j
                        Is = ntuple(_ -> rand(rng, 1:256, n), M)
                        resize!(p, n)
                        algorithm2_2!(p, Is, ws)
                        @test all(!iszero, p)
                        @test sum(p) ≈ 1.0
                    end
                end
            end
        end
    end
end

@testset "algorithm 3" begin
    w = [0.0, 10.0, 5.0, 0.0, 2.5]
    p = [0.25, 0.2857142857142857, 0.14285714285714285, 0.25, 0.07142857142857142]
    p′ = algorithm3(w, 0.5)
    @test sum(p′) ≈ 1.0
    @test p ≈ p′
    @test algorithm3(w, 1.0) == [0.5, 0.0, 0.0, 0.5, 0.0]
    # u not 0 ≤ u ≤ 1
    p = [0.75, -0.2857142857142857, -0.14285714285714285, 0.75, -0.07142857142857142]
    @test_throws DomainError algorithm3(w, 1.5)
    @test_throws DomainError algorithm3(w, -0.5)
    # @test p ≈ p′
    # @test sum(p′) ≈ 1.0
    # throws where appropriate
    @test_throws DimensionMismatch algorithm3!(p, zeros(6), 0.5)
    @test_throws DimensionMismatch algorithm3!(zeros(6), p, 0.5)
    # zeros behavior
    p = algorithm3(w, 0.0)
    @test p == [0.0, 0.5714285714285714, 0.2857142857142857, 0.0, 0.14285714285714285]
    algorithm3!(p, 1.0)
    @test p == [0.5, 0.0, 0.0, 0.5, 0.0]
    algorithm3!(p, 0.0)
    @test p == [0.5, 0.0, 0.0, 0.5, 0.0]
    p .= 0
    algorithm3!(p, 1.0)
    @test p == fill(0.2, 5)
    p .= 0
    algorithm3!(p, 0.0)
    @test all(p .=== -NaN)
    #
    w = zeros(5)
    for u ∈ (nextfloat(0.0), eps(), 0.5, prevfloat(1.0), 1.0)
        algorithm3!(p, w, u)
        @test p == fill(0.2, 5)
    end

    @testset "NaN handling (lack thereof)" begin
        # things which generate NaNs
        w = [0.0, 10.0, 5.0, 0.0, NaN]
        p = [0.25, NaN, NaN, 0.25, NaN]
        p′ = algorithm3(w, 0.5)
        @test isequal(p′, p)
        @test p′[1] + p′[4] == 0.5

        w = [1.0, 2.0, 3.0, NaN]
        p′ = algorithm3(w, 0.5)
        @test isequal(p′, fill(NaN, 4))
    end
    @testset "±Inf handling (lack thereof)" begin
        w = [0.0, 10.0, 5.0, 0.0, Inf]
        p = [0.25, 0.0, 0.0, 0.25, NaN]
        p′ = algorithm3(w, 0.5)
        @test isequal(p′, p)
        @test sum(p′[1:4]) == 0.5
        w = [0.0, 10.0, 5.0, 0.0, -Inf]
        p = [0.25, -0.0, -0.0, 0.25, -NaN]
        p′ = algorithm3(w, 0.5)
        @test all(p′ .=== p)
        @test sum(p′[1:4]) == 0.5

        w = [1.0, 2.0, 3.0, Inf]
        p = [0.0, 0.0, 0.0, NaN]
        p′ = algorithm3(w, 0.5)
        @test isequal(p′, p)

        # Inf * inv(Inf) -> NaN
        w = [0.0, 0.0, 0.0, Inf]
        p′ = algorithm3(w, 0.5)
        @test sum(p′[1:3]) ≈ 0.5
        @test isnan(p′[4]) # usually === -NaN, but maybe not always
        w = [0.0, 0.0, 0.0, -Inf]
        p′ = algorithm3(w, 0.5)
        @test sum(p′[1:3]) ≈ 0.5
        @test isnan(p′[4]) # usually === -NaN, but maybe not always
    end

    @testset "algorithm 3, sweep precision (SIMD rounding)" begin
        rng = Xoshiro(1234)
        w = rand(rng, 256)
        w[rand(rng, 1:256, 64)] .= 0
        x = copyto!(similar(w), w)
        p = similar(w)
        for u = 0.000001:.000001:0.999999
            w .= x .* u
            algorithm3!(p, w, u)
            @test all(!iszero, p)
            @test sum(p) ≈ 1.0
        end
        # When u = 1.0, all non-zero elements become zero
        u = 1.0
        w .= x .* u
        algorithm3!(p, w, u)
        @test count(iszero, p) == 197 # PRNG initialized as above gives 59 non-zeros
        @test sum(p) ≈ 1.0
        # When u = 0.0, all non-zero elements are normalized but zero elements remain same
        u = 0.0
        w .= x
        algorithm3!(p, w, u)
        @test count(iszero, p) == 59
        @test sum(p) ≈ 1.0
        # Large values -- potentially sensitive to very small u's around 1e-6
        for u = 0.0001:0.0001:0.9999
            for i = 1:20
                n = 1 << i
                w .= x .* n
                algorithm3!(p, w, u)
                @test all(!iszero, p)
                @test sum(p) ≈ 1.0
            end
        end
    end

    @testset "algorithm 3, type handling" begin
        p = [5/9, 1/9, 3/9, 0, 0]
        p2 = [5//9, 1//9, 3//9, 0, 0]
        # conversions should work and be type stable throughout
        @inferred algorithm3!(p, 0.5)
        @inferred algorithm3!(p2, 1)
        # while Integer <: Real, it clearly cannot be normalized in a meaningful manner
        p3 = [5, 1, 3, 0, 0]
        @test_throws InexactError algorithm3!(p3, 0)
        @test_throws InexactError algorithm3!(p3, 1)
        @test_throws DomainError algorithm3!(p3, 999)
        #
        w = [5/9, 1/9, 3/9, 0, 0]
        w2 = [5//9, 1//9, 3//9, 0, 0]
        w3 = [5, 1, 3, 0, 0]
        p = [5/9, 1/9, 3/9, 0, 0]
        p2 = [5//9, 1//9, 3//9, 0, 0]
        p3 = [5, 1, 3, 0, 0]
        @inferred algorithm3!(p, w, 0.5)
        @inferred algorithm3!(p, w3, 0.5)
        @inferred algorithm3!(p2, w2, 0.5)
        @inferred algorithm3!(p2, w2, 1//2)

        @test_throws InexactError algorithm3!(p3, w, 1)
        @test_throws InexactError algorithm3!(p3, w3, 1)
        @test_throws InexactError algorithm3!(p3, w2, 1)
        @test_throws InexactError algorithm3!(p3, w2, 1)
        ps = (algorithm3(w3, 1//2), algorithm3(w3, 0.5), algorithm3(w3, 1), algorithm3(w3, 0))
        for T ∈ (Float16, Float32, Float64, BigFloat, Int8, Int16, Int32, Int64, BigInt, Rational{Int8}, Rational{Int16}, Rational{Int32}, Rational{Int64}, Rational{Int128}, Rational{BigInt})
            𝑤 = T.(w3)
            for (i, u) ∈ enumerate((1//2, 0.5, 1, 0))
                p = @inferred algorithm3(𝑤, u)
                @test p ≈ ps[i]
            end
        end
    end
end


@testset "normalize1" begin
    # An aberrant case
    w = [-1.0, 1.0, 1.0]
    p = normalize1(w)
    @test p == w
    @test sum(p) ≈ 1.0
    w = abs.(w)
    p = normalize1(w)
    @test p == [1/3, 1/3, 1/3]
    @test sum(p) ≈ 1.0
    # The usual things
    A = [1.0 2.0; 3.0 4.0]
    B = normalize1(A)
    @test sum(B) ≈ 1.0
    normalize1!(B)
    @test sum(B) ≈ 1.0
    A[1] = -1.0
    normalize1!(B, A)
    @test sum(B) ≈ 1.0
    @test any(≤(0.0), B)
    @testset "NaN handling (lack thereof)" begin
        # things which generate NaNs
        A = zeros(2,2)
        @test isequal(normalize1(A), fill(NaN, 2,2))
        # preexisting NaN(s)
        A = [0.0 NaN; 0.0 1.0]
        @test isequal(normalize1(A), fill(NaN, 2,2))
        A = fill(NaN, 2,2)
        @test isequal(normalize1(A), fill(NaN, 2,2))
        A = fill(-NaN, 2,2)
        B = normalize1(A)
        @test all(A .=== B)
        @test all(A .=== -NaN)
    end
    @testset "±Inf handling (lack thereof)" begin
        A = [Inf Inf; Inf Inf]
        @test isequal(normalize1(A), fill(NaN, 2,2))
        A = [0.0 Inf; 0.0 1.0]
        @test isequal(normalize1(A), [0.0 NaN; 0.0 0.0])
        A = [0.0 -Inf; 0.0 1.0]
        B = normalize1(A)
        @test isequal(B, [-0.0 NaN; -0.0 -0.0])
        @test isequal(B, [-0.0 -NaN; -0.0 -0.0])
        # is the negative bit on the NaN set? (it should be)
        @test B[1,2] === -NaN
        @test B[1,2] !== NaN
        # propagating negative bit
        normalize1!(A, B)
        @test all(A .=== -NaN)
    end

    # Sweeps: rounding errors, etc.
    rng = Xoshiro(0x434b089281805289)
    for i = 1:20
        for j = -1:1
            n = (1 << i) + j
            w = rand(rng, n)
            p = normalize1(w)
            @test all(!iszero, p)
            @test sum(p) ≈ 1.0
            @test p ≈ normalize1(p)
            # x = copyto!(similar(w), w)
            # for u = 0.0001:0.0001:0.9999
            #     w .= x .* u
            #     normalize1!(p, w)
            #     @test all(!iszero, p)
            #     @test sum(p) ≈ 1.0
            # end
        end
    end
    x = rand(rng, 256)
    x[rand(rng, 1:256, 64)] .= 0
    y = normalize1(x)
    @test count(iszero, x) == count(iszero, y)
    @test sum(y) ≈ 1.0
    w = similar(x)
    # Large values
    for i = 1:20
        n = 1 << i
        w .= x .* n
        normalize1!(y, w)
        @test sum(y) ≈ 1.0
        w[129:256] .*= n >> i
        normalize1!(y, w)
        @test sum(y) ≈ 1.0
    end
    # Small values
    nz = count(iszero, x)
    for u = 0.000001:0.000001:0.999999
        w .= x .* u
        normalize1!(y, w)
        @test sum(y) ≈ 1.0
        @test count(iszero, y) == nz
        # mixed with large
        w[129:256] .*= 10^6
        @test sum(y) ≈ 1.0
        @test count(iszero, y) == nz
    end
end


@testset "normalizations, 0 unchanged component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 4, 5]
    w₂ = [10, 20, 10, 20, 10]
    p = algorithm2_1(I′, w)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [1/7, 2/7, 1/7, 2/7, 1/7]
end
@testset "normalizations, 1 unchanged component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 4, 5]
    w₂ = [10, 20, 0, 20, 10]
    p = algorithm2_1(I′, w)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q[3] == p[3]
    @test q[1] + q[2] + q[4] + q[5] == p[1] + p[2] + p[4] + p[5]
end
@testset "normalizations, 2 unchanged components" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 4, 5]
    w₂ = [10, 20, 0, 0, 10]
    p = algorithm2_1(I′, w)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q[3] == p[3] && q[4] == p[4]
    @test q[1] + q[2] + q[5] == p[1] + p[2] + p[5]
end
@testset "normalizations, all unchanged components" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 4, 5]
    w₂ = [0, 0, 0, 0, 0]
    p = algorithm2_1(I′, w)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test all(q .== p)
end
@testset "normalizations, 0 unchanged component, 0 unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 4, 5]
    u = 1/2
    w₂ = [10, 20, 10, 20, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [1/7, 2/7, 1/7, 2/7, 1/7]
end
@testset "normalizations, 0 unchanged component, 1 unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 4, 0]
    u = 1/2
    w₂ = [10, 20, 10, 20, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [1/7, 2/7, 1/7, 2/7, 1/7]
end
@testset "normalizations, 0 unchanged component, 2 unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 0, 0]
    u = 1/2
    w₂ = [10, 20, 10, 20, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [1/7, 2/7, 1/7, 2/7, 1/7]
end
@testset "normalizations, 0 unchanged component, all unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [0, 0, 0, 0, 0]
    u = 1/2
    w₂ = [10, 20, 10, 20, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [1/7, 2/7, 1/7, 2/7, 1/7]
end
@testset "normalizations, 1 unchanged component, 1 unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 4, 0]
    u = 1/2
    w₂ = [10, 20, 0, 20, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [0.14166666666666666, 0.2833333333333333, 0.15000000000000002, 0.2833333333333333, 0.14166666666666666]

    w₂ = [10, 20, 10, 20, 0]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q[5] == u
end
@testset "normalizations, 1 unchanged component, 2 unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 0, 0]
    u = 1/2
    w₂ = [10, 20, 0, 20, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [1/8, 1/4, 1/4, 1/4, 1/8]

    w₂ = [10, 20, 10, 20, 0]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q[5] == u / 2
    @test q == [1/8, 1/4, 1/8, 1/4, 1/4]
end
@testset "normalizations, 2 unchanged component, 2 unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 0, 0]
    u = 1/2
    w₂ = [10, 20, 0, 0, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == [1/8, 1/4, 1/4, 1/4, 1/8]

    w₂ = [10, 20, 10, 0, 0]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q[5] == u / 2
    @test q == [1/8, 1/4, 1/8, 1/4, 1/4]

    w₂ = [0, 20, 10, 20, 0]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q[5] == u / 2
    @test q == [0.16666666666666666, 0.2333333333333333, 0.11666666666666665, 0.2333333333333333, 0.25]
end
@testset "normalizations, all unchanged component, 2 unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [2, 1, 3, 0, 0]
    u = 1/2
    w₂ = [0, 0, 0, 0, 0]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q == p
end
@testset "normalizations, 2 unchanged component, all unknown component" begin
    I′ = [1, 2, 3, 4, 5]
    w = [0, 0, 0, 0, 0]
    u = 1/2
    w₂ = [10, 20, 0, 0, 10]
    p = algorithm2_1_algorithm3(I′, w, u)
    @test sum(p) ≈ 1
    q = algorithm4(p, w₂)
    @test sum(q) ≈ 1
    @test q ≈ [0.15, 0.3, 0.2, 0.2, 0.15]
end
# @testset "Monte Carlo, re-weighted: 1 unchanged component" begin
#     I′ = [1, 2, 3, 4, 5]
#     w = [2, 1, 3, 4, 5]
#     w₂ = [10, 20, 0, 20, 10]
#     A = weightedmcadd1(Int, (I′, w₂), w, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)
# end
# @testset "Monte Carlo, re-weighted: all unchanged component" begin
#     I′ = [1, 2, 3, 4, 5]
#     w = [2, 1, 3, 4, 5]
#     w₂ = [0, 0, 0, 0, 0]
#     A = weightedmcadd1(Int, (I′, w₂), w, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)
# end
# @testset "Monte Carlo, re-weighted: 1 unchanged component, 0 unknown component" begin
#     I′ = [1, 2, 3, 4, 5]
#     w = [2, 1, 3, 4, 5]
#     w₂ = [10, 20, 0, 20, 10]
#     u = 1/2
#     A = weightedmcadd1(Int, (I′, w₂), w, u, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w, u)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)
# end
# @testset "Monte Carlo, re-weighted: 1 unchanged component, 1 unknown component" begin
#     I′ = [1, 2, 3, 4, 5]
#     w = [2, 1, 3, 4, 0]
#     w₂ = [10, 20, 0, 20, 10]
#     u = 1/2
#     A = weightedmcadd1(Int, (I′, w₂), w, u, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w, u)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)

#     w = [2, 1, 0, 4, 5]
#     A = weightedmcadd1(Int, (I′, w₂), w, u, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w, u)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)
# end
# @testset "Monte Carlo, re-weighted: 1 unchanged component, 2 unknown component" begin
#     I′ = [1, 2, 3, 4, 5]
#     w = [2, 1, 3, 0, 0]
#     w₂ = [10, 20, 0, 20, 10]
#     u = 1/2
#     A = weightedmcadd1(Int, (I′, w₂), w, u, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w, u)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)

#     w = [2, 1, 0, 4, 0]
#     A = weightedmcadd1(Int, (I′, w₂), w, u, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w, u)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)
# end
# @testset "Monte Carlo, re-weighted: 1 unchanged component, all unknown component" begin
#     I′ = [1, 2, 3, 4, 5]
#     w = [0, 0, 0, 0, 0]
#     w₂ = [10, 20, 0, 20, 10]
#     u = 1/2
#     A = weightedmcadd1(Int, (I′, w₂), w, u, 10)
#     Σ = sum(A, dims=1)
#     @test all(==(1), Σ)
#     weightedmcadd1!(A, (I′, w₂), w, u)
#     Σ = sum(A, dims=1)
#     @test all(==(2), Σ)
# end
@testset "algorithm4 behavior" begin
    @test isequal(algorithm4(zeros(3), zeros(3)), [NaN, NaN, NaN])
    @test !isequal(algorithm4(rand(3), zeros(3)), [NaN, NaN, NaN])
    @test isequal(algorithm4(zeros(3), rand(3)), [NaN, NaN, NaN])
end
@testset "algorithm4, type handling" begin
    w₁ = [1, 1, 1, 1, 0]
    w₂ = [2, 1, 3, 0, 5]
    p = [3/22, 3/44, 9/44, 1/4, 15/44]
    pᵣ = [3//22, 3//44, 9//44, 1//4, 15//44]
    @test algorithm4(w₁, w₂) ≈ p
    for T ∈ (Float32, Float64, BigFloat, Rational{Int16}, Rational{Int32}, Rational{Int64}, Rational{Int128}, Rational{BigInt})
        𝑤₁ = T.(w₁)
        @test algorithm4(𝑤₁, w₂) ≈ p rtol=√(eps())
        𝑤₂ = T.(w₂)
        @test algorithm4(w₁, 𝑤₂) ≈ p rtol=√(eps())
        𝑝 = @inferred algorithm4(𝑤₁, 𝑤₂)
        @test eltype(𝑝) === T
        @test 𝑝 ≈ p rtol=∛(eps())
    end
    ################
    # Checks and corresponding benchmarks. Use @code_warntype and inspect the type of
    # p[i] = w₂ᵢ == zero(U) ? c₁ * w₁ᵢ : c₂ * w₂ᵢ
    # _v2! : c₁ = one(S)/ s₁; c₂ = s₁′ * c₁ / s₂
    #### 2-arg
    # 𝑤₁ = Rational.(w₁)
    # 𝑤₂ = Rational.(w₂)
    # 𝐰₁ = Float64.(w₁)
    # 𝐰₂ = Float64.(w₂)
    # @code_warntype algorithm4!(p, w₁)
    # @code_warntype algorithm4!(p, 𝑤₁)
    # @code_warntype algorithm4!(pᵣ, w₁)
    # # c₁::Rational{Int}, c₂::Float64
    # @code_warntype algorithm4!(pᵣ, 𝐰₁)

    # @code_warntype algorithm4_v2!(p, w₁)
    # @code_warntype algorithm4_v2!(p, 𝑤₁)
    # @code_warntype algorithm4_v2!(pᵣ, w₁)
    # # c₁::Rational{Int}, c₂::Float64
    # @code_warntype algorithm4_v2!(pᵣ, 𝐰₁)

    # @benchmark algorithm4!(p, w₁)
    # @benchmark algorithm4!(p, 𝑤₁)
    # @benchmark algorithm4!(pᵣ, w₁)
    # @benchmark algorithm4!(pᵣ, 𝐰₁)

    # @benchmark algorithm4_v2!(p, w₁)
    # @benchmark algorithm4_v2!(p, 𝑤₁)
    # @benchmark algorithm4_v2!(pᵣ, w₁)
    # @benchmark algorithm4_v2!(pᵣ, 𝐰₁)
    # #### 3-arg
    # @code_warntype algorithm4!(p, w₁, w₂)
    # @code_warntype algorithm4!(p, 𝑤₁, w₂)
    # # s₁::Float64, s₂::Rational{Int}
    # @code_warntype algorithm4!(p, w₁, 𝑤₂)
    # #
    # @code_warntype algorithm4!(pᵣ, w₁, w₂)
    # @code_warntype algorithm4_v2!(pᵣ, w₁, w₂)
    # #
    # @code_warntype algorithm4!(pᵣ, 𝑤₁, w₂)
    # @code_warntype algorithm4_v2!(pᵣ, 𝑤₁, w₂)
    # # c₁::Float64, c₂::Rational{Int}
    # @code_warntype algorithm4!(pᵣ, w₁, 𝑤₂)
    # @code_warntype algorithm4_v2!(pᵣ, w₁, 𝑤₂)
    # #
    # @benchmark algorithm4!($pᵣ, $𝑤₁, $w₂)
    # @benchmark algorithm4_v2!($pᵣ, $𝑤₁, $w₂)
    # #
    # @benchmark algorithm4!($pᵣ, $w₁, $𝑤₂)
    # @benchmark algorithm4_v2!($pᵣ, $w₁, $𝑤₂)
    # #
    # @benchmark algorithm4!($pᵣ, $𝑤₁, $𝑤₂)
    # @benchmark algorithm4_v2!($pᵣ, $𝑤₁, $𝑤₂)
    # #
    # @benchmark algorithm4!($p, $𝑤₁, $w₂)
    # @benchmark algorithm4_v2!($p, $𝑤₁, $w₂)
    # #
    # @benchmark algorithm4!($p, $w₁, $𝑤₂)
    # @benchmark algorithm4_v2!($p, $w₁, $𝑤₂)
    # #
    # @benchmark algorithm4!($p, $𝑤₁, $𝑤₂)
    # @benchmark algorithm4_v2!($p, $𝑤₁, $𝑤₂)
    # ####
    # @code_warntype algorithm4!(p, 𝐰₁, 𝑤₂)
    # #
    # @code_warntype algorithm4!(pᵣ, 𝐰₁, 𝐰₂)
    # @code_warntype algorithm4_v2!(pᵣ, 𝐰₁, 𝐰₂)
    # #
    # @code_warntype algorithm4!(pᵣ, 𝑤₁, 𝐰₂)
    # @code_warntype algorithm4_v2!(pᵣ, 𝑤₁, 𝐰₂)
    # # s₁::Float64, s₂::Rational{Int}
    # @code_warntype algorithm4!(pᵣ, 𝐰₁, 𝑤₂)
    # @code_warntype algorithm4_v2!(pᵣ, 𝐰₁, 𝑤₂)
    # #
    # @benchmark algorithm4!($pᵣ, $𝑤₁, $𝐰₂)
    # @benchmark algorithm4_v2!($pᵣ, $𝑤₁, $𝐰₂)
    # #
    # @benchmark algorithm4!($pᵣ, $𝐰₁, $𝑤₂)
    # @benchmark algorithm4_v2!($pᵣ, $𝐰₁, $𝑤₂)
    # #
    # @benchmark algorithm4!($pᵣ, $𝐰₁, $𝐰₂)
    # @benchmark algorithm4_v2!($pᵣ, $𝐰₁, $𝐰₂)
    # #
    # @benchmark algorithm4!($p, $𝑤₁, $𝐰₂)
    # @benchmark algorithm4_v2!($p, $𝑤₁, $𝐰₂)
    # #
    # @benchmark algorithm4!($p, $𝐰₁, $𝑤₂)
    # @benchmark algorithm4_v2!($p, $𝐰₁, $𝑤₂)
    # #
    # @benchmark algorithm4!($p, $𝐰₁, $𝐰₂)
    # @benchmark algorithm4_v2!($p, $𝐰₁, $𝐰₂)
    # #
end
@testset "algorithm3, algorithm4, application order effects" begin
    # 3 -> 4, w₁ ∌ 0, w₂ ∋ 0
    w₁ = [1., 2, 3, 4, 5]
    w₂ = [2, 1, 3, 4, 0]
    u = 0.5
    ω₁ = algorithm3(w₁, u)
    @test ω₁ ≈ w₁ ./ sum(w₁)
    ω = algorithm4(ω₁, w₂)
    @test sum(ω) ≈ 1
    @test ω[5] == ω₁[5]
    @test ω ≉ algorithm4(rand(5), w₂)
    # 3 -> 4, w₁ ∋ 0, w₂ ∌ 0
    w₁ = [1., 2, 3, 4, 0]
    w₂ = [2, 1, 3, 4, 5]
    u = 0.5
    ω₁ = algorithm3(w₁, u)
    @test sum(ω₁) ≈ 1
    @test 0 ∉ ω₁
    @test ω₁[5] == u
    ω = algorithm4(ω₁, w₂)
    @test sum(ω) ≈ 1
    @test ω ≈ algorithm4(rand(5), w₂)
    # 3 -> 4, w₁ ∌ 0, w₂ ∌ 0
    w₁ = [1., 2, 3, 4, 5]
    w₂ = [2, 1, 3, 4, 1]
    u = 0.5
    ω₁ = algorithm3(w₁, u)
    @test ω₁ ≈ w₁ ./ sum(w₁)
    ω = algorithm4(ω₁, w₂)
    @test sum(ω) ≈ 1
    @test ω[5] ≉ ω₁[5]
    # 3 -> 4, w₁ ∋ 0, w₂ ∋ 0
    # sub-case 1: J₁ ∩ I₂′ = ∅
    w₁ = [1., 2, 3, 4, 0]
    w₂ = [2, 1, 3, 4, 0]
    u = 0.5
    ω₁ = algorithm3(w₁, u)
    @test sum(ω₁) ≈ 1
    @test 0 ∉ ω₁
    ω = algorithm4(ω₁, w₂)
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test sum(ω) ≈ 1
    @test isdisjoint(J₁, I₂′)
    @test ω[5] == ω₁[5]
    @test ω ≉ algorithm4(rand(5), w₂)
    # sub-case 2: J₁ ∩ I₂′ ≠ ∅
    w₁ = [1., 2, 3, 4, 0]
    w₂ = [2, 1, 0, 0, 5]
    u = 0.5
    ω₁ = algorithm3(w₁, u)
    @test sum(ω₁) ≈ 1
    @test 0 ∉ ω₁
    ω = algorithm4(ω₁, w₂)
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test sum(ω) ≈ 1
    @test !isdisjoint(J₁, I₂′)
    @test ω[3] == ω₁[3] && ω[4] == ω₁[4]
    ####
    # 4 -> 3, w₁ ∌ 0, w₂ ∋ 0
    # J₁′ = ∅, J₂ ≠ ∅, thus, some elements reweighted (i.e. ∈ I₂′)
    w₁ = [1., 2, 3, 4, 5]
    w₂ = [2, 1, 3, 4, 0]
    u = 0.5
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁[5] == w₁[5] / sum(w₁)
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω == ω₁
    # 4 -> 3, w₁ ∋ 0, w₂ ∌ 0
    w₁ = [1., 2, 3, 4, 0]
    w₂ = [2, 1, 3, 4, 1]
    u = 0.5
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁[5] ≉ w₁[5] / sum(w₁)
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω == ω₁
    # 4 -> 3, w₁ ∌ 0, w₂ ∌ 0
    w₁ = [1., 2, 3, 4, 5]
    w₂ = [2, 1, 3, 4, 1]
    u = 0.5
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁ ≉ w₁ ./ sum(w₁)
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω == ω₁
    # 4 -> 3, w₁ ∋ 0, w₂ ∋ 0
    # sub-case 1: J₁ ∩ J₂ ≠ ∅, J₁ ∩ I₂′ = ∅
    # elements ∈ J₁ ∩ J₂ are remain zero after application of 4,
    # no zero elements become non-zero as J₁ ∩ I₂′ = ∅
    w₁ = [1., 2, 3, 4, 0]
    w₂ = [2, 1, 3, 4, 0]
    u = 0.5
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test !isdisjoint(J₁, J₂)
    @test isdisjoint(J₁, I₂′)
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁[5] == 0.0
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁ ≈ w₂ ./ sum(w₂)
    @test sum(ω₁[J₂]) ≈ sum(w₁[J₂]) / sum(w₁[I₁′])
    @test sum(w₁[J₂]) / sum(w₁[I₂′]) ≈ sum(ω₁[J₂]) / sum(ω₁[I₂′])
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω[5] == u
    # sub-case 2: J₁ ∩ J₂ = ∅, J₁ ∩ I₂′ ≠ ∅
    # no zero elements preserved on application of 4.
    w₁ = [1., 2, 3, 4, 0]
    w₂ = [2, 1, 3, 0, 5]
    u = 0.5
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test isdisjoint(J₁, J₂)
    @test !isdisjoint(J₁, I₂′)
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁[4] == w₁[4] / sum(w₁)
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁ ≉ w₂ ./ sum(w₂)
    @test sum(ω₁[J₂]) ≈ sum(w₁[J₂]) / sum(w₁[I₁′])
    @test sum(w₁[J₂]) / sum(w₁[I₂′]) ≈ sum(ω₁[J₂]) / sum(ω₁[I₂′])
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω == ω₁
    # sub-case 3: J₁ ∩ J₂ ≠ ∅, |J₁| > |J₁ ∩ J₂|, J₁ ∩ I₂′ ≠ ∅
    # elements ∈ J₁ ∩ I₂′ become non-zero
    # elements J₁ ∖ I₂′ = J₁ ∩ J₂ remain the same on application of 4
    w₁ = [1., 2, 3, 0, 0]
    w₂ = [2, 1, 3, 4, 0]
    u = 0.5
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test !isdisjoint(J₁, J₂)
    @test !isdisjoint(J₁, I₂′)
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁[5] == 0.0
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁ ≈ w₂ ./ sum(w₂)
    @test sum(ω₁[J₂]) ≈ sum(w₁[J₂]) / sum(w₁[I₁′])
    @test sum(w₁[J₂]) / sum(w₁[I₂′]) ≈ sum(ω₁[J₂]) / sum(ω₁[I₂′])
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω[5] == u
    # sub-case 4: J₁ ∩ J₂ ≠ ∅, |J₂| > |J₁ ∩ J₂|, J₂ ⊇ J₁, J₁ ∩ I₂′ = ∅
    # J₁ ∩ J₂ remain zero
    w₁ = [1., 2, 0, 4, 0]
    w₂ = [0, 1, 0, 4, 0]
    u = 0.5
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test !isdisjoint(J₁, J₂)
    @test isdisjoint(J₁, I₂′)
    @test !isdisjoint(J₂, I₁′)
    @test J₂ ⊇ J₁
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁[3] == ω₁[5] == 0.0
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁ ≉ w₂ ./ sum(w₂)
    @test sum(ω₁[J₂]) ≈ sum(w₁[J₂]) / sum(w₁[I₁′])
    @test sum(w₁[J₂]) / sum(w₁[I₂′]) ≈ sum(ω₁[J₂]) / sum(ω₁[I₂′])
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω[3] == ω[5] == u / 2
    # sub-case 5: J₁ ∩ J₂ ≠ ∅, |J₂| > |J₁ ∩ J₂|, J₂ ⊉ J₁, J₁ ∩ I₂′ ≠ ∅
    # elements ∈ J₁ ∩ I₂′ become non-zero
    # J₁ ∩ J₂ remain zero
    w₁ = [1., 2, 0, 4, 0]
    w₂ = [0, 0, 2, 4, 0]
    u = 0.5
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test !isdisjoint(J₁, J₂)
    @test !isdisjoint(J₁, I₂′)
    @test !isdisjoint(J₂, I₁′)
    @test J₂ ⊉ J₁
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁[5] == 0.0
    @test ω₁[3] != 0.0
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁ ≉ w₂ ./ sum(w₂)
    @test sum(ω₁[J₂]) ≈ sum(w₁[J₂]) / sum(w₁[I₁′])
    @test sum(w₁[J₂]) / sum(w₁[I₂′]) ≈ sum(ω₁[J₂]) / sum(ω₁[I₂′])
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω[5] == u
    # sub-case 6: J₁ ∩ J₂ ≠ ∅, |J₂| ≯ |J₁ ∩ J₂|, J₂ ⊉ J₁, J₁ ∩ I₂′ ≠ ∅
    # elements ∈ J₁ ∩ I₂′ become non-zero
    # elements ∈ (J₂ ∩ I₁′) ∪ (J₁ ∩ I₂′) are affected
    # J₁ ∩ J₂ remain zero
    # elements ∈ J₂ ∩ I₁′ become ωᵢ = w₁ᵢ / sum(w₁)
    w₁ = [1., 2, 0, 4, 0]
    w₂ = [2, 0, 3, 0, 0]
    u = 0.5
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test !isdisjoint(J₁, J₂)
    @test !isdisjoint(J₁, I₂′)
    @test !isdisjoint(J₂, I₁′)
    @test J₂ ⊉ J₁
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁[2] == w₁[2] / sum(w₁)
    @test ω₁[4] == w₁[4] / sum(w₁)
    @test ω₁[3] != 0.0
    @test ω₁[5] == 0.0
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁ ≉ w₂ ./ sum(w₂)
    @test sum(ω₁[J₂]) ≈ sum(w₁[J₂]) / sum(w₁[I₁′])
    @test sum(w₁[J₂]) / sum(w₁[I₂′]) ≈ sum(ω₁[J₂]) / sum(ω₁[I₂′])
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω[5] == u
    # sub-case 7: J₁ ∩ J₂ ≠ ∅, |J₁| > |J₁ ∩ J₂|, J₁ ⊇ J₁, J₁ ∩ I₂′ ≠ ∅, J₂ ∩ I₁′ = ∅
    # Essentially, w₂ overwrites w₁, because it re-weights all of the probability mass
    # elements ∈ J₁ ∩ I₂′ become non-zero
    w₁ = [1., 2, 0, 0, 0]
    w₂ = [5, 1, 3, 0, 0]
    u = 0.5
    J₁ = findall(iszero, w₁)
    J₂ = findall(iszero, w₂)
    I₁′ = findall(!iszero, w₁)
    I₂′ = findall(!iszero, w₂)
    @test !isdisjoint(J₁, J₂)
    @test !isdisjoint(J₁, I₂′)
    @test isdisjoint(J₂, I₁′)
    @test J₁ ⊇ J₁
    ω₁ = algorithm4(w₁, w₂)
    @test sum(ω₁) ≈ 1
    @test ω₁[3] != 0.0
    @test ω₁[4] == 0.0
    @test ω₁[5] == 0.0
    @test ω₁ ≉ w₁ ./ sum(w₁)
    @test ω₁ ≈ w₂ ./ sum(w₂)
    @test sum(ω₁[J₂]) ≈ sum(w₁[J₂]) / sum(w₁[I₁′])
    @test sum(w₁[J₂]) / sum(w₁[I₂′]) ≈ sum(ω₁[J₂]) / sum(ω₁[I₂′])
    ω = algorithm3(ω₁, u)
    @test sum(ω) ≈ 1
    @test ω[4] == u / 2
    @test ω[5] == u / 2
end
