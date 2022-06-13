#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################
# mirror of sampler.jl; separate file for variants on threading

function tsample(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, num_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    tsample!(B, A)
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    tsample!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, keep::NTuple{N, Bool}, default::NTuple{N, Int}, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L)
        U = Vector{Float64}(undef, L)
        Σω = Vector{T}()
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for (Iₛ, ω) ∈ a
                resize!(Σω, length(ω))
                cumsum!(Σω, ω)
                categorical!(C, U, Σω)
                @simd for l ∈ eachindex(C, 𝒥)
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

# A simplification: an array of sparse vectors
function tsample(::Type{S}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, num_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    tsample!(B, A)
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    tsample!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, keep::NTuple{N, Bool}, default::NTuple{N, Int}, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L)
        U = Vector{Float64}(undef, L)
        Σω = Vector{T}()
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            Iₛ, ω = A[IA]
            resize!(Σω, length(ω))
            cumsum!(Σω, ω)
            categorical!(C, U, Σω)
            @simd for l ∈ eachindex(C, 𝒥)
                c = C[l]
                j = 𝒥[l]
                B[Iₛ[c], j, IR] += one(S)
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



# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function tsample(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, num_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    tsample!(B, A)
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}) where {S<:Real, N′} where {N}
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    tsample!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, keep::NTuple{N, Bool}, default::NTuple{N, Int}, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L) # similar(𝒥, Int)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
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
            Threads.@spawn tsample!(B, A, keep, default, start:h)
            tsample!(B, A, keep, default, (h + 1):stop)
        end
        return B
    end
end

# A simplification: an array of sparse vectors
function tsample(::Type{S}, A::AbstractArray{Vector{Int}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, num_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    tsample!(B, A)
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}) where {S<:Real, N′} where {N}
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    tsample!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}, keep::NTuple{N, Bool}, default::NTuple{N, Int}, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            Iₛ = A[IA]
            rand!(C, Iₛ)
            @simd for l ∈ eachindex(C, 𝒥)
                c = C[l]
                j = 𝒥[l]
                B[c, j, IR] += one(S)
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