#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

# Examples: equal probability mass
A = [[[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]]
B = sample(Int, A, 10, 6, (1,))
B′ = dropdims(B, dims=2)
@code_warntype sample!(B, A, (1,))

A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
B = sample(Int, A, 10, 6, (1,))
B′ = dropdims(B, dims=2)
@code_warntype sample!(B, A, (1,))

C = fill(A, 2,3,4);
B = sample(Int, C, 10, 6, (1,3));
@code_warntype sample!(B, C, (1,3))

################

B_1 = sample(Int, A, 1000, 6, (1,));
B_2 = sample_simd(Int, A, 6, 1000);

@benchmark sample!($B_1, $A, $(1,))
@benchmark sample_simd!($B_2, $A)

# Eventually, at num_categories=10^4, num_samples=10^5, the in-order traversal wins
B_3 = sample(Int, C, 1000, 6, (1,2,3));
B_4 = sample_simd(Int, C, 6, 1000);

@benchmark sample!($B_3, $C, $(1,2,3))
@benchmark sample_simd!($B_4, $C)

using Random
A′ = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
A′ = [[1, 1000], [100, 200, 300, 400], [200, 400, 600, 800, 1000, 900]]
# A′ = [[1, 10], [10, 20, 30, 40], [20, 40, 60, 80, 100, 900]]
D′ = fill(A′, 100,50,50);

n_sim = 10^3
@timev B_5 = sample1(Int, D′, n_sim, num_cat(D′), (1,2,3));
@timev B_6 = sample_simd(Int, D′, num_cat(D′), n_sim);
@timev B_7 = sample2(Int, D′, n_sim, num_cat(D′), (1,));
@timev B_7_3 = sample3(Int, D′, n_sim, num_cat(D′), (1,));


@benchmark sample!($B_5, $D′, $(1,2,3))
@benchmark sample_simd!($B_6, $D′)
@benchmark sample2!($B_7, $D′, $(1,2,3))

n_sim = 10^4
@timev B_8 = tsample1(Int, D′, n_sim, num_cat(D′), (1,2,3));
@timev B_9 = tsample_simd(Int, D′, num_cat(D′), n_sim);
@timev B_10 = tsample2(Int, D′, n_sim, num_cat(D′), (1,2,3));
@timev B_11 = tsample3(Int, D′, n_sim, num_cat(D′), (1,2,3));
@timev B_12 = tsample2(Int, D′, 1000, num_cat(D′), (1,));
sum(B_8) == sum(B_9) == sum(B_10)

################################################################

# Examples: unequal probability mass
A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
# A = [([1, 1000], [0.3, 0.7]), ([100,200,300,400], [0.2, 0.3, 0.4, 0.1]), ([200, 400, 600, 800, 1000, 900], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]

D = fill(A, 100,50,50);

@timev B = sample(Int, A, 1000, num_cat(A), (1,));
@timev B′ = sample(Int, A′, 1000, num_cat(A), (1,));
@timev sample!(B, A, (1,))

n_sim = 10^3
dims = (1,2,3)
@timev B_1 = sample(Int, D, n_sim, num_cat(D), dims);
@timev sample!(B_1, D);
@code_warntype sample!(B_1, D)

@timev B_2 = sample(Int, D′, n_sim, num_cat(D′), dims);
@timev sample!(B_2, D′);
@code_warntype sample!(B_2, D′)


# @timev B_1_4 = sample4(Int, D, n_sim, num_cat(D), dims);
# @timev B_2_4 = sample4(Int, D′, n_sim, num_cat(D′), dims);
# sum(B_1) == sum(B_2) == sum(B_1_4) == sum(B_2_4)
# @timev sum(B_1, dims=2);
# @timev sum(B_1_4, dims=1);

@timev B_3 = tsample(Int, D, 100000, num_cat(D), (1,2,3));
@timev B_4 = tsample(Int, D′, 100000, num_cat(D), (1,2,3));


function countcategory(A::AbstractArray{T, N}) where {T<:Integer, N}
    mx = maximum(A)
    v = zeros(Int, mx)
    @inbounds @simd for i ∈ eachindex(A)
        v[A[i]] += 1
    end
    v
end

################################################################
# Limiting chunksize of U; single sparse vectors.

# Equal probability mass
A = [1,2,3,4,5,6]
n_sim = 10^3
B = sample(Int, A, n_sim, num_cat(A), (1,));
@benchmark sample!($B, $A)
@benchmark sample0!($B, $A)
@benchmark sample2!($B, $A)
@code_warntype sample!(B, A)
@code_warntype sample2!(B, A)
sum(B)
@timev sample!(B, A)
@timev sample2!(B, A)

ω = [0.1, 0.1, 0.1, 0.1,0.1, 0.5]
Σω = cumsum(ω)

# Unequal probability mass
A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
# A = ([1,2,3,4,5,6], [0.1, 0.25, 0.05, 0.25,0.15, 0.2])
n_sim = 10^3
B = sample(Int, A, n_sim, num_cat(A), (1,));
@benchmark sample!($B, $A)
@benchmark sample0!($B, $A)
@benchmark sample2!($B, $A)
@code_warntype sample!(B, A)
@code_warntype sample2!(B, A)
sum(B)
@timev sample!(B, A)
@timev sample2!(B, A)

