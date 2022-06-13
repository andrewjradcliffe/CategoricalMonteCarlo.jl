#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################
# mirror of sampler.jl; separate file for variants on threading

function tsample(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, num_samples, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    tsample!(B, A, dims)
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    tsample!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L) # similar(𝒥, Int)
        U = Vector{Float64}(undef, L)
        Σω = Vector{T}()
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for (Iₛ, ω) ∈ a
                resize!(Σω, length(ω))
                cumsum!(Σω, ω)
                categorical!(C, U, Σω)
                for l ∈ eachindex(C, 𝒥)
                # @simd for (c, j) ∈ zip(C, 𝒥)
                    c = C[l]
                    j = 𝒥[l]
                    B[Iₛ[c], j, IR] += one(S)
                end
            end
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample!(B, A, keep, default, start:h)
            tsample!(B, A, keep, default, (h + 1):stop)
        end
        return B
    end
end


function tsample(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_samples::Int, num_categories::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_categories, num_samples, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    tsample!(B, A, dims)
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}) where {S<:Real, N′} where {N}
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    tsample!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L) # similar(𝒥, Int)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for Iₛ ∈ a
                rand!(C, Iₛ)
                for l ∈ eachindex(C, 𝒥)
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
            Threads.@spawn tsample!(B, A, keep, default, start:h)
            tsample!(B, A, keep, default, (h + 1):stop)
        end
        return B
    end
end
