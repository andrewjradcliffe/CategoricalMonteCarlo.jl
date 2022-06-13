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

# The bare minimum for `sample` interface-- covers all 4 other definitions.

sample(::Type{S}, A, n_sim, n_cat; dims=:) where {S} = sample(S, A, n_sim, n_cat, dims)
sample(::Type{S}, A, n_sim; dims=:) where {S} = sample(S, A, n_sim, num_cat(A), dims)
sample(::Type{S}, A, n_sim::Int, n_cat::Int, dims::Int) where {S} = sample(S, A, n_sim, n_cat, (dims,))

function sample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    sample!(B, A)
end

function sample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T, N}
    B = fill!(similar(A, S, (n_cat, n_sim)), zero(S))
    sample!(B, A)
end

# # The expected case: vectors of sparse vectors (as their bare components)
# function sample(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
#     Dᴬ = size(A)
#     Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
#     B = fill!(similar(A, S, Dᴮ), zero(S))
#     sample!(B, A)
# end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    U = Vector{Float64}(undef, size(B, 2))
    Σω = Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, ω) ∈ a
            resize!(Σω, length(ω))
            cumsum!(Σω, ω)
            categorical!(C, U, Σω)
            @simd for j ∈ axes(B, 2)
                c = C[j]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

# # A simplification: an array of sparse vectors
# function sample(::Type{S}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
#     Dᴬ = size(A)
#     Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
#     B = fill!(similar(A, S, Dᴮ), zero(S))
#     sample!(B, A)
# end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    U = Vector{Float64}(undef, size(B, 2))
    Σω = Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, ω = A[IA]
        resize!(Σω, length(ω))
        cumsum!(Σω, ω)
        categorical!(C, U, Σω)
        @simd for j ∈ axes(B, 2)
            c = C[j]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

# # A simple check
# A = rand(2,3,4,5,6);
# N = ndims(A)
# dims = (2,3,5)
# n_cat = 3
# n_sim = 10

# keep = ntuple(d -> d ∉ dims, Val(N))
# default = ntuple(d -> firstindex(A, d), Val(N))

# Dᴬ = size(A)
# Dᴮ = tuple(n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., n_sim)
# B = similar(A, S, Dᴮ);
# keep_b, default_b = Broadcast.newindexer(B)
# keep_b[2:end-1] == keep

# # Specialized method for eltype(A)::Vector{Vector{Int}}
# # or, in other words, where the probability mass on each element is 1 / length(Iₛ)
# function sample(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
#     Dᴬ = size(A)
#     Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
#     B = fill!(similar(A, S, Dᴮ), zero(S))
#     sample!(B, A)
# end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
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

# # A simplification: an array of sparse vectors
# function sample(::Type{S}, A::AbstractArray{Vector{Int}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
#     Dᴬ = size(A)
#     Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
#     B = fill!(similar(A, S, Dᴮ), zero(S))
#     sample!(B, A)
# end

function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        @simd for j ∈ axes(B, 2)
            c = rand(Iₛ)
            B[c, j, IR] += one(S)
        end
    end
    B
end

# Conveniences
num_cat(A::AbstractArray{Vector{T}, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}}, N} =
    maximum(a -> maximum(((I, w),) -> maximum(I, init=0), a, init=0), A, init=0)
num_cat(A::AbstractArray{T, N}) where {T<:Tuple{Vector{Int}, Vector{<:AbstractFloat}} ,N} =
    maximum(((I, w),) -> maximum(I, init=0), A, init=0)

num_cat(A::AbstractArray{Vector{Vector{Int}}, N}) where {N} = maximum(a -> maximum(x -> maximum(x, init=0), a, init=0), A, init=0)
num_cat(A::AbstractArray{Vector{Int}, N}) where {N} = maximum(x -> maximum(x, init=0), A, init=0)

@noinline function _check_reducedims(B, A)
    Rdims = axes(B)[3:end]
    n_cat = num_cat(A)
    n_cat′ = length(axes(B, 1))
    n_cat ≤ n_cat′ || throw(DimensionMismatch("cannot sample from $(n_cat) categories into array with $(n_cat′) categories"))
    length(Rdims) ≤ ndims(A) || throw(DimensionMismatch("cannot reduce $(ndims(A))-dimensional array to $(length(Rdims)) trailing dimensions"))
    for i ∈ eachindex(Rdims)
        Ri, Ai = Rdims[i], axes(A, i)
        length(Ri) == 1 || Ri == Ai || throw(DimensionMismatch("reduction on array with indices $(axes(A)) with output with indices $(Rdims)"))
    end
    true
end
