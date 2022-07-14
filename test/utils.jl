# Test of utilities

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
    #
    B0 = Int[]
    B1 = zeros(Int, 1,2)
    B2 = zeros(Int, 2,2)
    for prototype ∈ ([1, 2], ([1, 2], [0.4, 0.6]), [0.4, 0.6], SparseVector([0.0, 1.0]))
        @test_throws DimensionMismatch _check_reducedims(B0, prototype)
        @test_throws DimensionMismatch _check_reducedims(B0, [prototype])
        @test_throws DimensionMismatch _check_reducedims(B0, [[prototype]])
        @test_throws DimensionMismatch _check_reducedims(B1, prototype)
        @test_throws DimensionMismatch _check_reducedims(B1, [prototype])
        @test_throws DimensionMismatch _check_reducedims(B1, [[prototype]])
        @test _check_reducedims(B2, prototype)
        @test _check_reducedims(B2, [prototype])
        @test _check_reducedims(B2, [[prototype]])
    end
end

@testset "num_cat" begin
    A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    @test num_cat(A) == 6
    A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
    @test num_cat(A) == 6
    A = [1,2,3,4,5,6]
    @test num_cat(A) == 6
    A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
    @test num_cat(A) == 6
    A = [[([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]]
    @test num_cat(A) == 6
    A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
    @test num_cat(A) == 6
    A = [[[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]]
    @test num_cat(A) == 6
    A = [[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]
    @test num_cat(A) == 6
    A = [0.1, 0.1, 0.1, 0.1,0.1, 0.5]
    @test num_cat(A) == 6
    sv1 = SparseVector(2, [1,2], [0.3, 0.7])
    sv2 = SparseVector(4, [1,2,3,4], [0.2, 0.3, 0.4, 0.1])
    sv3 = SparseVector(6, [1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
    A = [[sv1, sv2, sv3]]
    @test num_cat(A) == 6
    A = [sv1, sv2, sv3]
    @test num_cat(A) == 6
    @test num_cat(sv3) == 6
    # empty cases
    @test num_cat(Vector{Vector{Vector{Int}}}()) == 0
    @test num_cat(Vector{Vector{Int}}()) == 0
    @test num_cat(Vector{Int}()) == 0
    @test num_cat(Vector{Vector{Tuple{Vector{Int}, Vector{Float64}}}}()) == 0
    @test num_cat(Vector{Tuple{Vector{Int}, Vector{Float64}}}()) == 0
    @test num_cat((Vector{Int}(), Vector{Float64}())) == 0
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

@testset "splitranges" begin
    b = 16
    for a = -16:16
        ur = a:b
        for c = 1:b
            rs = splitranges(ur, c)
            @test sum(length, rs) == length(ur)
        end
    end
end

@testset "bounds_cat" begin
    A1 = [[-1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
    A2 = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 7]]
    B = [A1, A2]
    @test bounds_cat(A1) == (-1, 6)
    @test bounds_cat(A2) == (1, 7)
    @test bounds_cat(B) == (-1, 7)
    # emptys
    A3 = [Int[], Int[]]
    @test bounds_cat(A3) == (1, 0)
    B3 = [A3, A3]
    @test bounds_cat(B3) == (1, 0)
    B4 = [A3, A1]
    @test bounds_cat(B4) == (-1, 6)
    #
    A1 = [([-1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
    A2 = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,7], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
    B = [A1, A2]
    @test bounds_cat(A1) == (-1, 6)
    @test bounds_cat(A2) == (1, 7)
    @test bounds_cat(B) == (-1, 7)
    # emptys
    A3 = [(Int[], Float64[]), (Int[], Float64[])]
    @test bounds_cat(A3) == (1, 0)
    B3 = [A3, A3]
    @test bounds_cat(B3) == (1, 0)
    B4 = [A3, A1]
    @test bounds_cat(B4) == (-1, 6)
    #
    A1 = [[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]
    A2 = [[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1]]
    B = [A1, A2]
    @test bounds_cat(A1) == (1, 6)
    @test bounds_cat(A2) == (1, 5)
    @test bounds_cat(B) == (1, 6)
    @test bounds_cat(Float64[]) == (1, 0)
    @test bounds_cat([Float64[]]) == (1, 0)
    @test bounds_cat([[Float64[]]]) == (1, 0)
    A3 = [Float64[], Float64[]]
    @test bounds_cat(A3) == (1, 0)
    B3 = [A3, A3]
    @test bounds_cat(B3) == (1, 0)
    B4 = [A3, A1]
    @test bounds_cat(B4) == (1, 6)
    #
    x = SparseVector([0.0, 1.0, 2.0, 0.0, 0.0, 0.0])
    @test bounds_cat(x) == (1, 6)
    A = [x, SparseVector([0.0, 1.0, 2.0, 0.0]), SparseVector([1.0, 0.0])]
    @test bounds_cat(A) == (1, 6)
    @test bounds_cat([A, A]) == (1, 6)
    ####
    # An interesting case
    a1 = ([9999, 1439], [0.8029133268547554, 0.1970866731452445])
    a2 = ([9284, 4370, 2965, 1590], [0.10222319762724291, 0.13054189392858026, 0.43245627176252643, 0.3347786366816504])
    a3 = ([6289, 308, 6378, 7212, 5426, 662], [0.03053777422684849, 0.21452879865837565, 0.6835396454000753, 0.0713937817147005])
    B = [a1, a2, a3]
    A = first.(B)
    @test bounds_cat(B) == (308, 9999)
end
