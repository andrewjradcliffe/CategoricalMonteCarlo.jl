#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

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
    keeps = ntuple(d -> d ∈ dims, Val(N))
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
    keeps = ntuple(d -> d ∈ dims, Val(N))
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
