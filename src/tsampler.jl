#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################
# mirror of sampler.jl; separate file for variants on threading

# The bare minimum for `sample` interface-- covers all 4 other definitions.
tsample(::Type{S}, A, n_sim, n_cat; dims=:) where {S} = tsample(S, A, n_sim, n_cat, dims)
tsample(::Type{S}, A, n_sim; dims=:) where {S} = tsample(S, A, n_sim, num_cat(A), dims)
tsample(::Type{S}, A, n_sim::Int, n_cat::Int, dims::Int) where {S} = tsample(S, A, n_sim, n_cat, (dims,))

function tsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    tsample!(B, A)
end

function tsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T, N}
    B = fill!(similar(A, S, (n_cat, n_sim)), zero(S))
    tsample!(B, A)
end

# for recursive spawning
function tsample!(B, A)
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    tsample!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

# # The expected case: vectors of sparse vectors (as their bare components)
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
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
                for l ∈ eachindex(C, 𝒥)
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

# # A simplification: an array of sparse vectors
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
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
            for l ∈ eachindex(C, 𝒥)
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



# # Specialized method for eltype(A)::Vector{Vector{Int}}
# # or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L)
        @inbounds for IA ∈ CartesianIndices(A)
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

# # A simplification: an array of sparse vectors
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            Iₛ = A[IA]
            rand!(C, Iₛ)
            for l ∈ eachindex(C, 𝒥)
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
