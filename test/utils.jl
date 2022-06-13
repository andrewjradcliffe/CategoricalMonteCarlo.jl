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
