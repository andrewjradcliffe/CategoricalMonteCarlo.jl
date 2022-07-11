#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################
# mirror of sampler.jl; separate file for variants on threading

# The bare minimum for `sample` interface-- covers all 4 other definitions.
tsample(::Type{S}, A, n_sim, n_cat; dims=:, chunksize=5000) where {S} = tsample(S, A, n_sim, n_cat, dims, chunksize)
tsample(::Type{S}, A, n_sim; dims=:, chunksize=5000) where {S} = tsample(S, A, n_sim, num_cat(A), dims, chunksize)
tsample(::Type{S}, A, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S} = tsample(S, A, n_sim, n_cat, (dims,), chunksize)

function tsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}, chunksize::Int) where {S<:Real} where {P} where {T, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    tsample!(B, A, chunksize)
end

function tsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T, N}
    B = fill!(similar(A, S, (n_cat, n_sim)), zero(S))
    tsample!(B, A, chunksize)
end

tsample!(B, A; chunksize::Int=5000) = tsample!(B, A, chunksize)
function tsample!(B, A, chunksize::Int)
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        _sample_chunk!(B, A, keep, default, r)
    end
    return B
end

# The expected case: vectors of sparse vectors (as their bare components)
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    C, U = _genstorage_init(Float64, length(𝒥))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, p) ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            generate!(C, U, K, V)
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
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    L = length(𝒥)
    C, U = _genstorage_init(Float64, L)
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        generate!(C, U, K, V)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
tsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = tsample(S, A, n_sim, n_cat, :, chunksize)
tsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = tsample(S, A, n_sim, n_cat, :, chunksize)

function tsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A, chunksize)
end

function tsample!(B::AbstractMatrix, A::Tuple{Vector{Int}, Vector{<:AbstractFloat}}, chunksize::Int)
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        _sample_chunk!(B, A, r)
    end
    return B
end

function _sample_chunk!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    Iₛ, p = A
    K, V = sqhist(p)
    C = generate(K, V, length(𝒥))
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[Iₛ[c], j] += one(S)
    end
    return B
end
function _sample_chunk!(B::AbstractMatrix{S}, A::Tuple{AbstractVector{Int}, AbstractVector{T}}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    Iₛ, p = A
    n = length(Iₛ)
    Iₛp = (copyto!(Vector{Int}(undef, n), Iₛ), copyto!(Vector{T}(undef, n), p))
    _sample_chunk!(B, Iₛp, 𝒥)
end

################
# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    C, U = _genstorage_init(Float64, length(𝒥))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            n = length(Iₛ)
            generate!(C, U, n)
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
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    C, U = _genstorage_init(Float64, length(𝒥))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        n = length(Iₛ)
        generate!(C, U, n)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
function tsample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {N}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A, chunksize)
end

function tsample!(B::AbstractMatrix, A::Vector{Int}, chunksize::Int)
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        _sample_chunk!(B, A, r)
    end
    return B
end

function _sample_chunk!(B::AbstractMatrix{S}, A::AbstractVector{Int}, 𝒥::UnitRange{Int}) where {S<:Real}
    Iₛ = A
    n = length(Iₛ)
    C = generate(n, length(𝒥))
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[Iₛ[c], j] += one(S)
    end
    return B
end

################
# General case: dense vectors, the linear index of which indicates the category
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M}
    C, U = _genstorage_init(Float64, length(𝒥))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for p ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            generate!(C, U, K, V)
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
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{T}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    C, U = _genstorage_init(Float64, length(𝒥))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        generate!(C, U, K, V)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[c, j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a dense vector
tsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = tsample(S, A, n_sim, n_cat, :, chunksize)
tsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = tsample(S, A, n_sim, n_cat, :, chunksize)

function tsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A, chunksize)
end

function tsample!(B::AbstractMatrix, A::Vector{T}, chunksize::Int) where {T<:AbstractFloat}
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        _sample_chunk!(B, A, r)
    end
    return B
end

function _sample_chunk!(B::AbstractMatrix{S}, A::AbstractVector{T}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    p = copyto!(Vector{T}(undef, length(A)), A)
    K, V = sqhist(p)
    C = generate(K, V, length(𝒥))
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[c, j] += one(S)
    end
    return B
end

################
# General case: sparse vectors, the nzval of which indicates the category
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M}
    C, U = _genstorage_init(Float64, length(𝒥))
    K, V, q = _sqhist_init(Tv, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for sv ∈ a
            Iₛ, p = sv.nzind, sv.nzval
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            generate!(C, U, K, V)
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
function _sample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{SparseVector{Tv, Ti}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {Tv<:AbstractFloat, Ti<:Integer, N}
    C, U = _genstorage_init(Float64, length(𝒥))
    K, V, q = _sqhist_init(Tv, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        sv = A[IA]
        Iₛ, p = sv.nzind, sv.nzval
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        generate!(C, U, K, V)
        for l ∈ eachindex(C, 𝒥)
            c = C[l]
            j = 𝒥[l]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
tsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = tsample(S, A, n_sim, n_cat, :, chunksize)
tsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = tsample(S, A, n_sim, n_cat, :, chunksize)

function tsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A, chunksize)
end

function tsample!(B::AbstractMatrix, A::SparseVector{<:AbstractFloat}, chunksize::Int)
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 2):lastindex(B, 2), chunksize)
    @batch for r in rs
        _sample_chunk!(B, A, r)
    end
    return B
end

function _sample_chunk!(B::AbstractMatrix{S}, A::SparseVector{T}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    Iₛ, p = A.nzind, A.nzval
    K, V = sqhist(p)
    C = generate(K, V, length(𝒥))
    @inbounds for l ∈ eachindex(C, 𝒥)
        c = C[l]
        j = 𝒥[l]
        B[Iₛ[c], j] += one(S)
    end
    return B
end
