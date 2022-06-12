#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

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
# -- There is always the option of sampling across the j-indices of B
# and placing dimensions of A on the 3rd...end positions.
# If annotated with @inbounds and @simd, this is as fast (or faster) than
# the simple `sample_simd` approach.
function sample2(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, num_samples, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample2!(B, A, dims)
end

function sample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    C = Vector{Int}(undef, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keeps, defaults)
        a = A[IA]
        for Iₛ ∈ a
            rand!(C, Iₛ)
            @simd for j ∈ axes(B, 2)
                c = C[j]
                B[c, j, IR] += one(S)
            end
        end
    end
    B
end

function sample3(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_samples)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample3!(B, A, dims)
end

function sample3!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    C = Vector{Int}(undef, size(B, N′))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keeps, defaults)
        a = A[IA]
        for Iₛ ∈ a
            rand!(C, Iₛ)
            @simd for j ∈ axes(B, N′)
                c = C[j]
                B[c, IR, j] += one(S)
            end
        end
    end
    B
end

function tsample2(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, num_samples, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    tsample2!(B, A, dims)
end

function tsample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    tsample2!(B, A, keeps, defaults, firstindex(B, 2):size(B, 2))
end

function tsample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, keeps, defaults, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L) # similar(𝒥, Int)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keeps, defaults)
            a = A[IA]
            for Iₛ ∈ a
                rand!(C, Iₛ)
                @simd for l ∈ eachindex(C, 𝒥)
                    c = C[l]
                    j = 𝒥[l]
                    B[c, j, IR] += one(S)
                end
            end
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample2!(B, A, keeps, defaults, start:h)
            tsample2!(B, A, keeps, defaults, (h + 1):stop)
        end
        return B
    end
end

function tsample3(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_samples)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    tsample3!(B, A, dims)
end

function tsample3!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keeps = ntuple(d -> d ∉ dims, Val(N))
    defaults = ntuple(d -> firstindex(A, d), Val(N))
    tsample3!(B, A, keeps, defaults, firstindex(B, N′):size(B, N′))
end

function tsample3!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, keeps, defaults, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L) # similar(𝒥, Int)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keeps, defaults)
            a = A[IA]
            for Iₛ ∈ a
                rand!(C, Iₛ)
                @simd for l ∈ eachindex(C, 𝒥)
                    c = C[l]
                    j = 𝒥[l]
                    B[c, IR, j] += one(S)
                end
            end
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample3!(B, A, keeps, defaults, start:h)
            tsample3!(B, A, keeps, defaults, (h + 1):stop)
        end
        return B
    end
end

A = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
A = [[1, 1000], [100, 200, 300, 400], [200, 400, 600, 800, 1000, 900]]
D = fill(A, 100,50,50);

num_sim = 10^3
@timev B_5 = sample(Int, D, num_sim, num_cat(D), (1,2,3));
@timev B_6 = sample_simd(Int, D, num_cat(D), num_sim);
@timev B_7 = sample2(Int, D, num_sim, num_cat(D), (1,));
@timev B_7_3 = sample3(Int, D, num_sim, num_cat(D), (1,));


@benchmark sample!($B_5, $D, $(1,2,3))
@benchmark sample_simd!($B_6, $D)
@benchmark sample2!($B_7, $D, $(1,2,3))

num_sim = 10^5
@timev B_8 = tsample(Int, D, num_sim, num_cat(D), (1,2,3));
@timev B_9 = tsample_simd(Int, D, num_cat(D), num_sim);
@timev B_10 = tsample2(Int, D, num_sim, num_cat(D), (1,2,3));
@timev B_11 = tsample3(Int, D, num_sim, num_cat(D), (1,2,3));
@timev B_12 = tsample2(Int, D, 1000, num_cat(D), (1,));
sum(B_8) == sum(B_9) == sum(B_10)

function sample_simd!(B::Matrix{T}, A::Vector{Vector{Int}}) where {T<:Real}
    c = Vector{Int}(undef, size(B, 2))
    @inbounds for m ∈ eachindex(A)
        Iₛ = A[m]
        if length(Iₛ) == 1
            @inbounds i = Iₛ[1]
            @inbounds @simd ivdep for j ∈ axes(B, 2)
                B[i, j] += one(T)
            end
        else
            rand!(c, Iₛ)
            @inbounds @simd for j ∈ axes(B, 2)
                i = c[j]
                B[i, j] += one(T)
            end
        end
    end
    return B
end
sample_simd(::Type{T}, A::Vector{Vector{Int}}, I::Int, J::Int) where {T<:Real} =
    sample_simd!(zeros(T, I, J), A)

function sample_simd!(B::Matrix{T}, A::Array{Vector{Vector{Int}}, N}) where {T<:Real} where {N}
    c = Vector{Int}(undef, size(B, 2))
    @inbounds for n ∈ eachindex(A)
        a = A[n]
        for m ∈ eachindex(a)
            Iₛ = a[m]
            if length(Iₛ) == 1
                @inbounds i = Iₛ[1]
                @inbounds @simd ivdep for j ∈ axes(B, 2)
                    B[i, j] += one(T)
                end
            else
                rand!(c, Iₛ)
                @inbounds @simd for j ∈ axes(B, 2)
                    i = c[j]
                    B[i, j] += one(T)
                end
            end
        end
    end
    return B
end
sample_simd(::Type{T}, A::Array{Vector{Vector{Int}}, N}, I::Int, J::Int) where {T<:Real, N} =
    sample_simd!(zeros(T, I, J), A)


function tsample_simd!(B::Matrix{T}, A::Vector{Vector{Int}}, 𝒥::UnitRange{Int}) where {T<:Real}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        c = Vector{Int}(undef, L)
        @inbounds for m ∈ eachindex(A)
            Iₛ = A[m]
            if length(Iₛ) == 1
                @inbounds i = Iₛ[1]
                @inbounds @simd ivdep for j ∈ 𝒥
                    B[i, j] += one(T)
                end
            else
                rand!(c, Iₛ)
                @inbounds @simd for l ∈ eachindex(𝒥)
                    i = c[l]
                    B[i, 𝒥[l]] += one(T)
                end
            end
        end
        return B
    else
        H = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample_simd!(B, A, start:H)
            tsample_simd!(B, A, (H + 1):stop)
        end
        return B
    end
    return B
end
tsample_simd!(B::Matrix{T}, A::Vector{Vector{Int}}) where {T<:Real} = tsample_simd!(B, A, 1:size(B, 2))

tsample_simd(::Type{T}, A::Vector{Vector{Int}}, I::Int, J::Int) where {T<:Real} =
    tsample_simd!(zeros(T, I, J), A, 1:J)

function tsample_simd!(B::Matrix{T}, A::Array{Vector{Vector{Int}}, N}, 𝒥::UnitRange{Int}) where {T<:Real} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        c = Vector{Int}(undef, L)
        @inbounds for n ∈ eachindex(A)
            a = A[n]
            for m ∈ eachindex(a)
                Iₛ = a[m]
                if length(Iₛ) == 1
                    @inbounds i = Iₛ[1]
                    @inbounds @simd ivdep for j ∈ 𝒥
                        B[i, j] += one(T)
                    end
                else
                    rand!(c, Iₛ)
                    @inbounds @simd for l ∈ eachindex(𝒥)
                        i = c[l]
                        B[i, 𝒥[l]] += one(T)
                    end
                end
            end
        end
        return B
    else
        H = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample_simd!(B, A, start:H)
            tsample_simd!(B, A, (H + 1):stop)
        end
        return B
    end
end
tsample_simd!(B::Matrix{T}, A::Array{Vector{Vector{Int}}, N}) where {T<:Real} where {N} =
    tsample_simd!(B, A, 1:size(B, 2))
tsample_simd(::Type{T}, A::Array{Vector{Vector{Int}}, N}, I::Int, J::Int) where {T<:Real} where {N} = tsample_simd!(zeros(T, I, J), A, 1:J)
