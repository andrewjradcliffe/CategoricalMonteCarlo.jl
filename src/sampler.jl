#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

# A ∈ 𝔻ᴰ¹ˣᴰ²ˣᴰ³ˣ⋯ ; eltype(A) = Vector{Tuple{Vector{Int}, Vector{<:AbstractFloat}}}
#                                      (Iₛ, ω) OR (Iₛ, Σω)
# Each sampling routine is identical: unpack the tuple, draw c ~ Categorical(ω) and
# obtain the real category as Iₛ[c].
# This enables an encapsulation of all PVG-induced variability, hence, a consistent
# interface for the sampler.

# Technically, `sample` only needs to know ndims(A), not necessarily the element type.
# The appropriate dispatch on element type is necessary for `sample!`
# `sample` could instead use
# A::AbstractArray{U, N} where {U<:Union{Vector{Tuple{Vector{Int}, Vector{T}}}, Tuple{Vector{Int}, Vector{T}}}} where {T<:AbstractFloat}

# The expected case: vectors of sparse vectors (as their bare components)
function sample(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_samples)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample!(B, A, dims)
end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {T<:AbstractFloat, N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    for j ∈ axes(B, N′)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keeps, defaults)
            a = A[IA]
            for (Iₛ, ω) ∈ a
                c = categorical(ω)
                B[Iₛ[c], IR, j] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function sample(::Type{S}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_samples)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample!(B, A, dims)
end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {T<:AbstractFloat, N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    for j ∈ axes(B, N′)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keeps, defaults)
            Iₛ, ω = A[IA]
            c = categorical(ω)
            B[Iₛ[c], IR, j] += one(S)
        end
    end
    B
end


# # A simple check
# A = rand(2,3,4,5,6);
# N = ndims(A)
# dims = (2,3,5)
# num_categories = 3
# num_samples = 10

# keeps = ntuple(d -> d ∉ dims, Val(N))
# defaults = ntuple(d -> firstindex(A, d), Val(N))

# Dᴬ = size(A)
# Dᴮ = tuple(num_categories, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_samples)
# B = similar(A, S, Dᴮ);
# keeps_b, defaults_b = Broadcast.newindexer(B)
# keeps_b[2:end-1] == keeps





# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function sample(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_samples)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample!(B, A, dims)
end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    for j ∈ axes(B, N′)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keeps, defaults)
            a = A[IA]
            for Iₛ ∈ a
                c = rand(Iₛ)
                B[c, IR, j] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function sample(::Type{S}, A::AbstractArray{Vector{Int}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_samples)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample!(B, A, dims)
end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    for j ∈ axes(B, N′)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keeps, defaults)
            Iₛ = A[IA]
            c = rand(Iₛ)
            B[c, IR, j] += one(S)
        end
    end
    B
end

# Examples
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

# Conveniences
num_cat(A::AbstractArray{Vector{T}, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}}} =
    maximum(a -> maximum(((I, w),) -> maximum(I), a), A)
num_cat(A::AbstractArray{T, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}}} =
    maximum(((I, w),) -> maximum(I), A)

num_cat(A::AbstractArray{Vector{Vector{Int}}, N}) where {N} = maximum(a -> maximum(maximum, a), A)
num_cat(A::AbstractArray{Vector{Int}, N}) where {N} = maximum(maximum, A)


B_1 = sample(Int, A, 1000, 6, (1,));
B_2 = sample_simd(Int, A, 6, 1000);

@benchmark sample!($B_1, $A, $(1,))
@benchmark sample_simd!($B_2, $A)

# Eventually, at num_categories=10^4, num_samples=10^5, the in-order traversal wins
B_3 = sample(Int, C, 1000, 6, (1,2,3));
B_4 = sample_simd(Int, C, 6, 1000);

@benchmark sample!($B_3, $C, $(1,2,3))
@benchmark sample_simd!($B_4, $C)

# As the input array becomes large, SIMD PRNG sampling tends to be better
# due to the fact that each element of A is accessed only once.
D = fill(A, 100,50,50);

B_5 = sample(Int, D, 1000, 6, (1,2,3));
B_6 = sample_simd(Int, D, 6, 1000);

@benchmark sample!($B_5, $D, $(1,2,3))
@benchmark sample_simd!($B_6, $D)

function sample_simd!(B::Matrix{T}, A::Vector{Vector{Int}}) where {T<:Real}
    c = Vector{Int}(undef, size(A, 2))
    @inbounds for m ∈ eachindex(A)
        idxs = A[m]
        if length(idxs) == 1
            @inbounds i = idxs[1]
            @inbounds @simd ivdep for j ∈ axes(A, 2)
                B[i, j] += one(T)
            end
        else
            rand!(c, idxs)
            @inbounds @simd for j ∈ axes(A, 2)
                i = c[j]
                B[i, j] += one(T)
            end
        end
    end
    return A
end
sample_simd(::Type{T}, A::Vector{Vector{Int}}, I::Int, J::Int) where {T<:Real} =
    sample_simd!(zeros(T, I, J), A)

function sample_simd!(A::Matrix{T}, 𝓃A::Array{Vector{Vector{Int}}, N}) where {T<:Real} where {N}
    c = Vector{Int}(undef, size(A, 2))
    @inbounds for n ∈ eachindex(𝓃A)
        A = 𝓃A[n]
        for m ∈ eachindex(A)
            idxs = A[m]
            if length(idxs) == 1
                @inbounds i = idxs[1]
                @inbounds @simd ivdep for j ∈ axes(A, 2)
                    B[i, j] += one(T)
                end
            else
                rand!(c, idxs)
                @inbounds @simd for j ∈ axes(A, 2)
                    i = c[j]
                    B[i, j] += one(T)
                end
            end
        end
    end
    return A
end
sample_simd(::Type{T}, 𝓃A::Array{Vector{Vector{Int}}, N}, I::Int, J::Int) where {T<:Real, N} =
    sample_simd!(zeros(T, I, J), 𝓃A)
