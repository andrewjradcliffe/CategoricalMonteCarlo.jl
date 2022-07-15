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

n_sim = 10^4
dims = (1,2,3)
@timev B_1 = sample(Int, D, n_sim, num_cat(D), dims);
@timev sample!(B_1, D);
@code_warntype sample!(B_1, D)

@timev B_2 = sample(Int, D′, n_sim, num_cat(D′), dims);
@timev sample!(B_2, D′);
@code_warntype sample!(B_2, D′)
@timev vsample!(B_2, D′);
@timev sample_orderN!(B_2, D′);


A = [[0.3, 0.7], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1,0.1, 0.5]]
B = sample(Int, A, 10, num_cat(A), (1,))


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

#### actual SparseVector
using SparseArrays
Iₛ, ω = ([1,2,3,4], [0.2, 0.3, 0.4, 0.1])
sv = SparseVector(4, Iₛ, ω)
@timev (; n, nzind, nzval) = sv
sv1 = SparseVector(2, [1,2], [0.3, 0.7])
sv2 = SparseVector(4, [1,2,3,4], [0.2, 0.3, 0.4, 0.1])
sv3 = SparseVector(6, [1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
A = [sv1, sv2, sv3]
D = fill(A, 100,50,50);
@timev B = sample(Int, D, 1000, dims=(1,2,3));
@timev sample!(B, D);

# nzval must be in order to be a valid SparseVector
sv1 = SparseVector(1000, [1, 1000], [0.3, 0.7])
sv2 = SparseVector(400, [100,200,300,400], [0.2, 0.3, 0.4, 0.1])
sv3 = SparseVector(1000, [200, 400, 600, 800, 900, 1000], [0.1, 0.1, 0.1, 0.1,0.5, 0.1])
A = [sv1, sv2, sv3]
D = fill(A, 100,50,50);
@timev B = sample(Int, D, 1000, dims=(1,2,3));
@timev sample!(B, D);
@code_warntype sample!(B, D)

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

################
A = [1,2]
B = tsample(Int, A, 10^8);
@timev tsample!(B, A);
@timev tsample0!(B, A);
@timev sample!(B, A);

#### limiting chunksize, larger arrays
# Z = [[rand(1:1000, 5) for _ = 1:3] for _ = 1:50, _ = 1:50, _ = 1:50, _ = 1:10];
A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
Z = fill(A, 100,50,50);
n_sim = 10^4
@timev B = sample(Int, Z, n_sim, dims=(1,2,3));
# The smaller chunksize approach actually performs ≈5-8% worse.
@timev sample!(B, Z);
@timev sample2!(B, Z);


A = [([1, 2], [0.3, 0.7]), ([1,2,3,4], [0.2, 0.3, 0.4, 0.1]), ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])]
Z = fill(A, 100,50,50);
n_sim = 10^4
# The smaller chunksize approach actually performs ≈5-8% worse. -- true again for nonequiprobable
@timev B = sample(Int, Z, n_sim, dims=(1,2,3));
@timev sample!(B, Z);
@timev vsample!(B, Z);
@timev sample_orderN!(B, Z);

################################################################
# Marsaglia square histogram
A = ([1,2,3,4,5,6], [0.1, 0.1, 0.1, 0.1,0.1, 0.5])
n_sim = 10^3
B = sample(Int, A, n_sim, num_cat(A), (1,));
@benchmark sample!($B, $A)

@benchmark sample_mars!($B, $A)
@benchmark sample_mars2!($B, $A)
@benchmark sample_mars3!($B, $A)
@benchmark vsample!($B, $A)
#### Marsaglia, larger arrays
Z = fill(A, 100,50,50);
W = [Z Z Z Z];
n_sim = 10^4;
B = sample(Int, Z, n_sim, dims=:);
@timev sample!(B, Z);
@timev sample_mars!(B, Z);
@timev sample_mars2!(B, Z);
@timev vsample!(B, Z);
@timev vsample!(B, W);

E = zeros(Int, reverse(size(B)));
@timev sample_mars_dim1!(E, Z);
@timev sample_mars_dim1_4!(E, Z);

a = [1,2,3,4,5,6]
D′ = fill(a, 100,50,50);
B = zeros(6, 10000);
@benchmark sample_orderN!($B, $a)
@benchmark sample!($B, $a)
@benchmark vsample!($B, $a)
@benchmark sample!($B, $D′)
@benchmark vsample!($B, $D′)

@timev vtsample!(B, D′);
@timev tsample!(B, D′);
B′ = zeros(Int, 6, 10^6);
@timev vtsample!(B′, D′);
@timev tsample!(B′, D′);

# Threading with polyester
using Polyester
using VectorizationBase, Static
VectorizationBase.num_cores() = static(48)
B2 = zeros(Int, 6, 10^5);
@timev vtsample!(B2, D′, 10^4);
@timev vtsample!(B2, D′, 2 * 10^4);
@timev vtsample!(B′, D′, 2 * 10^4);


@timev vtsample!(B2, Z, chunksize=5000)
# 1.
using Random
using LoopVectorization
using Polyester
using SparseArrays
# 2. include's
# 3.
using VectorizationBase, Static
VectorizationBase.num_cores() = static(48)

B = zeros(Int, 6, 10^6);
v2 = [[1,2,3,4], [1,2,3,4,5,6]]
B2 =  vtsample(Int, v2, 10^4, chunksize=500)
@code_warntype vtsample!(B2, [[.5, .5], [.2, .8]], 500)
@timev vtsample(Int, [.5, .5], 10000, chunksize=500)
@timev vtsample(Int, [1,2], 10000, chunksize=500)

@timev vtsample!(B, D′, chunksize=10000);
@timev vtsample!(B, Z, chunksize=10000);

@timev vtsample!(B, D′, chunksize=1000);
@timev vtsample!(B, Z, chunksize=1000);

@timev vtsample!(B, D′, chunksize=100);
@timev vtsample!(B, Z, chunksize=100);
B .= 0;

sum(B, dims=2)

sum(length, Z) * 6 * size(B, 2)

################################################################
# Experiment with using a view to enable @turbo use everywhere
# Preliminary conclusion: not really much gain/loss in terms of time;
# the axes call on the view invokes some unnecessary allocations which can be
# alleviated by simply using axes(B, 2)
# Ultimately, it is likely not worth it to attempt it, as it is not really SIMD fodder
# due to the random nature of the indices; forcing SIMD will very likely
# make performance worse for any reasonable number of categories.
# Moreover, the random nature of the memory location being written to makes it an unsafe
# operation -- silent problems, but corrupted memory nonetheless.
## Addendum
# Memory would not be corrupted as each `j`-index is unique.

function vsample2!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        Bv = view(B, :, :, IR)
        for Iₛ ∈ a
            n = length(Iₛ)
            vgenerate!(C, U, n)
            @turbo for j ∈ indices((B, C), (2, 1))#axes(B, 2)#indices((Bv, C), (2, 1)) # axes(Bv, 2)
                c = C[j]
                Bv[Iₛ[c], j] += one(S)
            end
        end
    end
    B
end

function vsample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        Bv = view(B, :, :, IR)
        n = length(Iₛ)
        vgenerate!(C, U, n)
        @turbo for j ∈ axes(B, 2)
            c = C[j]
            Bv[Iₛ[c], j] += one(S)
        end
    end
    B
end

A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]];
D = fill(A, 10,10,10);
B = vsample(Int, D, 10000, dims=(1,3));


@benchmark vsample!($B, $D)
@benchmark vsample2!($B, $D)
@benchmark vsample3!($B, $D)

vsample2!(B, D);

@timev vsample!(B, Z);
@timev vsample2!(B, Z);

#
OB = OffsetArray(B, 1:6, 0:9999, 1:1, 1:1, 1:1);
vsample3!(OB, D);

OB2 = OffsetArray(B, 1:6, -5000:4999, 1:1, 1:1, 1:1);
vsample3!(OB2, D)

OB3 = OffsetArray(B, 1:6, 0:9999, 0:0, 0:0, 2:2);
vsample3!(OB3, D)

# much worse with @simd ivdep
function vsample3!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            n = length(Iₛ)
            vgenerate!(C, U, n)
            for (j′, j) ∈ enumerate(axes(B, 2))
                c = C[j′]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

function vsample3!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        n = length(Iₛ)
        vgenerate!(C, U, n)
        @simd ivdep for j ∈ eachindex(axes(B, 2), C)
            c = C[j]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

# Oddly, the fastest vsampler is non-allocating -- most likely due to
# the elimination of store + access instructions associated with using a temporary array.
function vsample2!(B::AbstractMatrix{S}, Iₛ::Vector{Int}) where {S<:Real}
    _check_reducedims(B, Iₛ)
    n = length(Iₛ)
    # C = vgenerate(n, size(B, 2))
    @inbounds for j ∈ axes(B, 2)
        c = generate(n)
        B[Iₛ[c], j] += one(S)
    end
    B
end

function vsample3!(B::AbstractMatrix{S}, Iₛ::Vector{Int}) where {S<:Real}
    _check_reducedims(B, Iₛ)
    n = length(Iₛ)
    # C = vgenerate(n, size(B, 2))
    @inbounds @simd ivdep for j ∈ axes(B, 2)
        c = generate(n)
        B[Iₛ[c], j] += one(S)
    end
    B
end

################
# Dimension experiments
#### Preliminary conclusion
# As expected, when the number of categories is small, i.e. 𝒪(1), simulation index being
# on the first dimension presents considerable advantages: ≈ 4x faster than by it being on
# the second dimension. However, as the number of categories increases to 𝒪(10) (and beyond),
# there is no difference in performance. Intuitively, this makes sense, as in the case of
# 𝒪(1) categories the instruction pipelining would be able to recognize that the same
# few chunks of memory are needed; even if it cannot figure it out, the same couple
# pieces of memory would always be in the cache simply on the basis of demand.
# as the number of categories increases, instruction pipelining would assuredly break down,
# and, regardless, there would be more variability in the memory demand. Consequently,
# this makes it increasingly unlikely that all require memory might be in the cache
# at any given time. This implies a scaling with cache size -- if cache permits it,
# then simulation index on first dimension will be superior. If not, then it's a wash.
# Notably, for most practical applications, number of categories is probably sufficient to
# make it a wash. Nonetheless, it is interesting to verify the theoretical prediction
# of superior performance when placing simulation index on first dimension.
#### A second look -- 1ˢᵗ dimension: simulation index, 2ⁿᵈ dimension: category index
## 1.
# There are advantages to placing simulations on the 1ˢᵗ dimension when considering
# scalar reductions, e.g. `+`, `max`, `min`; `mean`, `var` (variations on `mapreducethen`).
# However, if manipulating a whole simulation, then it is advantageous to place simulations
# on the 2ⁿᵈ dimension. Whole simulation transformations are more rare than scalar
# transformations/reductions. Notably, non-scalar transformations/reductions are almost
# inevitably bespoke affairs due to the difficult nature of a general programming model
# for non-scalar operations (`mapslices` exists, but `mapsubslices` does not).
# Thus, most users would invoke scalar operations, and for non-scalar, write their own.
# It seems important to have the scalar reductions be fast by default, rather than
# have the non-scalar be fast but rarely used.
# ∴ If non-scalar transformations/reductions are desired, the burden is on the user
# to implement them efficiently, either via permuting the dimensions, or some other scheme.
#     -- Another option is to provide both methods, as constructing the transpose
#        (1ˢᵗ: category, 2ⁿᵈ: simulation) directly may be useful. --
## 2.
# Placing simulation index on the 1ˢᵗ dimension and category index on the 2ⁿᵈ dimension
# also enables performance gains during sampling, if the data permit.
# Fundamentally, the performance gain during sampling with 1ˢᵗ: simulation, 2ⁿᵈ: category
# occurs due to the fact that for a subset of categories, fewer memory blocks are needed
# in the cache to complete 1,…,N simulations. Conversely, 1ˢᵗ: category, 2ⁿᵈ: simulation
# guarantees that at least one block must be loaded for each simulation -- a full
# traversal, with the block loaded on each simulation random by definition.
# What if the number of categories is large? (under 1ˢᵗ: sim, 2ⁿᵈ: cat)
#     - In the worse case, all categories are involved, and one is randomly loading
#       columns -- an equivalent situation to the default under 1ˢᵗ: cat, 2ⁿᵈ: sim.
#     - Even if the number of categories being sampled is large, if it is some subset
#       of the total, then less than a full traversal occurs.
#     ∴ any time n_cat < N_cat, it is better to have simulation index on the 1ˢᵗ dimension.
##
# Taken together, these two aspects provide strong motivation for 1ˢᵗ: sim, 2ⁿᵈ: cat.
################
function vsample2_dim1!(B::AbstractMatrix{S}, Iₛ::Vector{Int}) where {S<:Real}
    _check_reducedims(B, Iₛ)
    n = length(Iₛ)
    @inbounds for i ∈ axes(B, 1)
        c = generate(n)
        B[i, Iₛ[c]] += one(S)
    end
    B
end

function vsample3_dim1!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            n = length(Iₛ)
            vgenerate!(C, U, n)
            for (i′, i) ∈ enumerate(axes(B, 1))
                c = C[i′]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    B
end

function vsample3_dim1!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, p) ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            vgenerate!(C, U, K, V)
            for (i′, i) ∈ enumerate(axes(B, 1))
                c = C[i′]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    B
end

vtsample_dim1!(B, A; chunksize::Int=5000) = vtsample_dim1!(B, A, chunksize)
function vtsample_dim1!(B, A, chunksize::Int)
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    rs = splitranges(firstindex(B, 1):lastindex(B, 1), chunksize)
    @batch for r in rs
        _vsample_chunk_dim1!(B, A, keep, default, r)
    end
    return B
end

function _vsample_chunk_dim1!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    C, U = _genstorage_init(Float64, length(ℐ))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, p) ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            vgenerate!(C, U, K, V)
            for l ∈ eachindex(C, ℐ)
                c = C[l]
                i = ℐ[l]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    return B
end

n_cat = 10^3
D = [[rand(1:n_cat, 2), rand(1:n_cat, 4), rand(1:n_cat, 6)] for _ = 1:10, _ = 1:10, _ = 1:10];
E = [[(rand(1:n_cat, 2), normalize1!(rand(2))),
      (rand(1:n_cat, 4), normalize1!(rand(4))),
      (rand(1:n_cat, 6), normalize1!(rand(4)))] for _ = 1:10, _ = 1:10, _ = 1:10];
E2 = [[(rand(1:n_cat, 20), normalize1!(rand(20))),
       (rand(1:n_cat, 40), normalize1!(rand(40))),
       (rand(1:n_cat, 60), normalize1!(rand(40)))] for _ = 1:10, _ = 1:10, _ = 1:10];

# # Equal sized output.
# B = zeros(Int, n_cat,n_cat,1,1,1);
# @benchmark vsample!($B, $D)
# @benchmark vsample3_dim1!($B, $D)
# @benchmark vsample!($B, $E)
# @benchmark vsample3_dim1!($B, $E)
# @benchmark vsample!($B, $E2)
# @benchmark vsample3_dim1!($B, $E2)

# appropriately sized
n_sim = 10^3
B_dim2 = zeros(Int, n_cat, n_sim);
B_dim1 = zeros(Int, n_sim, n_cat);

@benchmark vsample!($B_dim2, $D)
@benchmark vsample3_dim1!($B_dim1, $D)
@benchmark vsample!($B_dim2, $E)
@benchmark vsample3_dim1!($B_dim1, $E)
@benchmark vsample!($B_dim2, $E2)
@benchmark vsample3_dim1!($B_dim1, $E2)

# larger simulation number
n_sim = 10^4
B_dim2 = zeros(Int, n_cat, n_sim);
B_dim1 = zeros(Int, n_sim, n_cat);

@benchmark vsample!($B_dim2, $D)
@benchmark vsample3_dim1!($B_dim1, $D)
@benchmark vsample!($B_dim2, $E)
@benchmark vsample3_dim1!($B_dim1, $E)
@benchmark vsample!($B_dim2, $E2)
@benchmark vsample3_dim1!($B_dim1, $E2)

# artificially inflated number of categories
n_cat = 10^4
B_dim2 = zeros(Int, n_cat, n_sim);
B_dim1 = zeros(Int, n_sim, n_cat);

@benchmark vsample!($B_dim2, $D)
@benchmark vsample3_dim1!($B_dim1, $D)
@benchmark vsample!($B_dim2, $E)
@benchmark vsample3_dim1!($B_dim1, $E)
@benchmark vsample!($B_dim2, $E2)
@benchmark vsample3_dim1!($B_dim1, $E2)

# large difference in tile size
n_cat, n_sim = 10^3, 10^5
B_dim2 = zeros(Int, n_cat, n_sim);
B_dim1 = zeros(Int, n_sim, n_cat);

@benchmark vsample!($B_dim2, $D)
@benchmark vsample3_dim1!($B_dim1, $D)
@benchmark vsample!($B_dim2, $E)
@benchmark vsample3_dim1!($B_dim1, $E)
@benchmark vsample!($B_dim2, $E2)
@benchmark vsample3_dim1!($B_dim1, $E2)

# no reduction
n_cat, n_sim = 10^3, 10^3
B_dim2 = zeros(Int, n_cat, n_sim, 10, 10, 10);
B_dim1 = zeros(Int, n_sim, n_cat, 10, 10, 10);

@benchmark vsample!($B_dim2, $D)
@benchmark vsample3_dim1!($B_dim1, $D)
@benchmark vsample!($B_dim2, $E)
@benchmark vsample3_dim1!($B_dim1, $E)
@benchmark vsample!($B_dim2, $E2)
@benchmark vsample3_dim1!($B_dim1, $E2)

# threading examples
n_cat, n_sim = 10^3, 10^5
B_dim2 = zeros(Int, n_cat, n_sim, 1, 1, 1);
B_dim1 = zeros(Int, n_sim, n_cat, 1, 1, 1);


@benchmark vtsample!($B_dim2, $D, chunksize=10000)
@benchmark vtsample_dim1!($B_dim1, $D, chunksize=10000)
@benchmark vtsample!($B_dim2, $E, chunksize=10000)
@benchmark vtsample_dim1!($B_dim1, $E, chunksize=10000)
@benchmark vtsample!($B_dim2, $E2, chunksize=10000)
@benchmark vtsample_dim1!($B_dim1, $E2, chunksize=10000)

B_dim2 .= 0;
B_dim1 .= 0;

vtsample!(B_dim2, E2, chunksize=10000);
vtsample_dim1!(B_dim1, E2, chunksize=10000);


# @turbo on the additions: small speed gain, but more temporaries -- probably not worth
# it given the potential problems with non-unit strides when working across multiple
# dimensions.
B = B_dim2;
keep, default = Broadcast.shapeindexer(axes(B)[3:end])
rs = splitranges(firstindex(B, 2):lastindex(B, 2), 10000);
𝒥 = rs[1]

B .= 0;

@benchmark _vsample_chunk!($B_dim2, $E, $keep, $default, $𝒥)
@benchmark _vsample_chunk2!($B_dim2, $E, $keep, $default, $𝒥)
@benchmark _vsample_chunk!($B_dim2, $E2, $keep, $default, $𝒥)
@benchmark _vsample_chunk2!($B_dim2, $E2, $keep, $default, $𝒥)

@benchmark _vsample_chunk_dim1!($B_dim1, $E, $keep, $default, $𝒥)
@benchmark _vsample_chunk_dim1_2!($B_dim1, $E, $keep, $default, $𝒥)
@benchmark _vsample_chunk_dim1!($B_dim1, $E2, $keep, $default, $𝒥)
@benchmark _vsample_chunk_dim1_2!($B_dim1, $E2, $keep, $default, $𝒥)

function _vsample_chunk2!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    C, U = _genstorage_init(Float64, length(𝒥))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        Bv = view(B, :, 𝒥, IR)
        for (Iₛ, p) ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            vgenerate!(C, U, K, V)
            @turbo for j ∈ indices((Bv, C), (2, 1))
                c = C[j]
                Bv[Iₛ[c], j] += one(S)
            end
        end
    end
    return B
end


function _vsample_chunk_dim1_2!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    C, U = _genstorage_init(Float64, length(ℐ))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        Bv = view(B, ℐ, :, IR)
        for (Iₛ, p) ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            vgenerate!(C, U, K, V)
            @turbo for i ∈ indices((Bv, C), (1, 1))
                c = C[i]
                Bv[i, Iₛ[c]] += one(S)
            end
        end
    end
    return B
end
