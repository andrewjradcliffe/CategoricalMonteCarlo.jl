#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################
# mirror of sampler.jl; separate file for variants on threading

# The bare minimum for `sample` interface-- covers all 4 other definitions.
vtsample(::Type{S}, A, n_sim, n_cat; dims=:, chunksize=5000) where {S} = vtsample(S, A, n_sim, n_cat, dims, chunksize)
vtsample(::Type{S}, A, n_sim; dims=:, chunksize=5000) where {S} = vtsample(S, A, n_sim, num_cat(A), dims, chunksize)
vtsample(::Type{S}, A, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S} = vtsample(S, A, n_sim, n_cat, (dims,), chunksize)

function vtsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}, chunksize::Int) where {S<:Real} where {P} where {T, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    vtsample!(B, A, chunksize)
end

function vtsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T, N}
    B = fill!(similar(A, S, (n_cat, n_sim)), zero(S))
    vtsample!(B, A, chunksize)
end

vtsample!(B, A; chunksize::Int=5000) = vtsample!(B, A, chunksize)
function vtsample!(B, A, chunksize::Int)
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        vsample_chunk!(B, A, keep, default, r)
    end
    return B
end

# The expected case: vectors of sparse vectors (as their bare components)
function vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init(T)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, ω) ∈ a
            n = length(ω)
            resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
            marsaglia!(K, V, q, ix, ω)
            vmarsaglia_generate!(C, U, K, V)
            for l ∈ eachindex(C, 𝒥)
                c = C[l]
                j = 𝒥[l]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of sparse vectors
function vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init(T)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, ω = A[IA]
        n = length(ω)
        resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
        marsaglia!(K, V, q, ix, ω)
        vmarsaglia_generate!(C, U, K, V)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
vtsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = vtsample(S, A, n_sim, n_cat, :, chunksize)
vtsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = vtsample(S, A, n_sim, n_cat, :, chunksize)

function vtsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix, A::Tuple{Vector{Int}, Vector{<:AbstractFloat}}, chunksize::Int)
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        vsample_chunk!(B, A, r)
    end
    return B
end

function vsample_chunk!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    L = length(𝒥)
    Iₛ, ω = A
    K, V = marsaglia(ω)
    C = vmarsaglia_generate!(Vector{Int}(undef, L), K, V)
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[Iₛ[c], j] += one(S)
    end
    return B
end

################
# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init()
    ω = Vector{Float64}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            n = length(Iₛ)
            resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
            resize!(ω, n)
            fill!(ω, inv(n))
            marsaglia!(K, V, q, ix, ω)
            vmarsaglia_generate!(C, U, K, V)
            for l ∈ eachindex(C, 𝒥)
                c = C[l]
                j = 𝒥[l]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of sparse vectors
function vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init()
    ω = Vector{Float64}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        n = length(Iₛ)
        resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
        resize!(ω, n)
        fill!(ω, inv(n))
        marsaglia!(K, V, q, ix, ω)
        vmarsaglia_generate!(C, U, K, V)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
function vtsample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {N}
    B = zeros(S, n_cat, n_sim)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix, A::Vector{Int}, chunksize::Int)
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        vsample_chunk!(B, A, r)
    end
    return B
end

function vsample_chunk!(B::AbstractMatrix{S}, A::AbstractVector{Int}, 𝒥::UnitRange{Int}) where {S<:Real}
    L = length(𝒥)
    Iₛ = A
    n = length(Iₛ)
    K, V = marsaglia(fill(inv(n), n))
    C = vmarsaglia_generate!(Vector{Int}(undef, L), K, V)
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[Iₛ[c], j] += one(S)
    end
    return B
end

################
# General case: dense vectors, the linear index of which indicates the category
function vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init(T)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for ω ∈ a
            n = length(ω)
            resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
            marsaglia!(K, V, q, ix, ω)
            vmarsaglia_generate!(C, U, K, V)
            for l ∈ eachindex(C, 𝒥)
                c = C[l]
                j = 𝒥[l]
                B[c, j, IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of dense vectors
function vtsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{T}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init(T)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        ω = A[IA]
        n = length(ω)
        resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
        marsaglia!(K, V, q, ix, ω)
        vmarsaglia_generate!(C, U, K, V)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[c, j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a dense vector
vtsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = vtsample(S, A, n_sim, n_cat, :, chunksize)
vtsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = vtsample(S, A, n_sim, n_cat, :, chunksize)

function vtsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix, A::Vector{<:AbstractFloat}, chunksize::Int)
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        vsample_chunk!(B, A, r)
    end
    return B
end

function vsample_chunk!(B::AbstractMatrix{S}, A::Vector{T}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    L = length(𝒥)
    ω = A
    K, V = marsaglia(ω)
    C = vmarsaglia_generate!(Vector{Int}(undef, L), K, V)
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[c, j] += one(S)
    end
    return B
end


################
# General case: sparse vectors, the nzval of which indicates the category
function vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init(Tv)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for sv ∈ a
            (; n, nzind, nzval) = sv
            Iₛ, ω = nzind, nzval
            n = length(ω)
            resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
            marsaglia!(K, V, q, ix, ω)
            vmarsaglia_generate!(C, U, K, V)
            for l ∈ eachindex(C, 𝒥)
                c = C[l]
                j = 𝒥[l]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of sparse vectors
function vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{SparseVector{Tv, Ti}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {Tv<:AbstractFloat, Ti<:Integer, N}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, ix, q = _marsaglia_init(Tv)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        sv = A[IA]
        (; n, nzind, nzval) = sv
        Iₛ, ω = nzind, nzval
        n = length(ω)
        resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
        marsaglia!(K, V, q, ix, ω)
        vmarsaglia_generate!(C, U, K, V)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
vtsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = vtsample(S, A, n_sim, n_cat, :, chunksize)
vtsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = vtsample(S, A, n_sim, n_cat, :, chunksize)

function vtsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix, A::SparseVector{<:AbstractFloat}, chunksize::Int)
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        vsample_chunk!(B, A, r)
    end
    return B
end

function vsample_chunk!(B::AbstractMatrix{S}, A::SparseVector{T}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    L = length(𝒥)
    (; n, nzind, nzval) = A
    Iₛ, ω = nzind, nzval
    K, V = marsaglia(ω)
    C = vmarsaglia_generate!(Vector{Int}(undef, L), K, V)
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[Iₛ[c], j] += one(S)
    end
    return B
end
