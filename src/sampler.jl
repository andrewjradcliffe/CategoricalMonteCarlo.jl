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

# Actually, this should ideally support:

# array of array of (sparse) vector
# A::AbstractArray{T, N} where {N} where {T<:AbstractArray{S, M}} where {M} where {S<:Union{Vector{Int}, Tuple{Vector{Int}, Vector{<:AbstractFloat}}}}

# array of (sparse) vector
# A::AbstractArray{T, N} where {N} where {T<:Union{Vector{Int}, Tuple{Vector{Int}, Vector{<:AbstractFloat}}}}

# (sparse) vector
# A::Union{Vector{Int}, Tuple{Vector{Int}, Vector{<:AbstractFloat}}}

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
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
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
            for j ∈ axes(B, 2)
                c = C[j]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

# # A simplification: an array of sparse vectors
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
        for j ∈ axes(B, 2)
            c = C[j]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

# # The simplest case: a sparse vector
sample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = sample(S, A, n_sim, n_cat, :)
sample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = sample(S, A, n_sim, n_cat, :)

function sample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat} where {N}
    B = zeros(S, n_cat, n_sim)
    sample!(B, A)
end

function sample!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    Iₛ, ω = A
    k = length(ω)
    Σω = cumsum(ω)
    s₀ = Σω[1]
    @inbounds for j ∈ axes(B, 2)
        u = rand()
        c = 1
        s = s₀
        while s < u && c < k
            c += 1
            s = Σω[c]
        end
        B[Iₛ[c], j] += one(S)
    end
    B
end

################
# # Specialized method for eltype(A)::Vector{Vector{Int}}
# # or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            rand!(C, Iₛ)
            for j ∈ axes(B, 2)
                c = C[j]
                B[c, j, IR] += one(S)
            end
        end
    end
    B
end

# # A simplification: an array of sparse vectors
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        rand!(C, Iₛ)
        for j ∈ axes(B, 2)
            c = C[j]
            B[c, j, IR] += one(S)
        end
    end
    B
end

# # The simplest case: a sparse vector
function sample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {N}
    B = zeros(S, n_cat, n_sim)
    sample!(B, A)
end

# Oddly, the fastest sampler is non-allocating -- most likely due to
# the elimination of store + access instructions associated with using a temporary.
function sample!(B::AbstractMatrix{S}, A::Vector{Int}) where {S<:Real}
    _check_reducedims(B, A)
    @inbounds for j ∈ axes(B, 2)
        c = rand(A)
        B[c, j] += one(S)
    end
    B
end
