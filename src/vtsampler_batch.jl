#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################
# mirror of sampler.jl; separate file for variants on threading

# The bare minimum for `sample` interface-- covers all 4 other definitions.
vtsample(::Type{S}, A, n_sim; dims=:, n_cat=nothing, chunksize=5000) where {S<:Real} = _vtsample(S, A, n_sim, n_cat, dims, chunksize)
vtsample(A, n_sim; dims=:, n_cat=nothing, chunksize=5000) = _vtsample(Int, A, n_sim, n_cat, dims, chunksize)

_vtsample(::Type{S}, A, n_sim, n_cat::Int, dims::Int, chunksize) where {S<:Real} = _vtsample(S, A, n_sim, n_cat, (dims,), chunksize)
_vtsample(::Type{S}, A, n_sim, ::Nothing, dims, chunksize) where {S<:Real} = _vtsample(S, A, n_sim, num_cat(A), dims, chunksize)

function _vtsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}, chunksize::Int) where {S<:Real} where {P} where {T, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_sim, n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    vtsample!(B, A, chunksize)
end

function _vtsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T, N}
    B = fill!(similar(A, S, (n_sim, n_cat)), zero(S))
    vtsample!(B, A, chunksize)
end

vtsample!(B, A; chunksize::Int=5000) = vtsample!(B, A, chunksize)
function vtsample!(B, A, chunksize::Int)
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    rs = splitranges(firstindex(B, 1):lastindex(B, 1), chunksize)
    @batch for r in rs
        _vsample_chunk!(B, A, keep, default, r)
    end
    return B
end

################
# The expected case: vectors of sparse vectors (as their bare components)
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    C, U = _genstorage_init(Float64, length(ℐ))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, p) ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            vgenerate!(C, U, K, V)
            for l ∈ eachindex(C, ℐ)
                c = C[l]
                i = ℐ[l]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of sparse vectors
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    L = length(ℐ)
    C, U = _genstorage_init(Float64, L)
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        vgenerate!(C, U, K, V)
        for l ∈ eachindex(C, ℐ)
            c = C[l]
            i = ℐ[l]
            B[i, Iₛ[c], IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
_vtsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
_vtsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
function _vtsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_sim, n_cat)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{<:AbstractFloat}}, chunksize::Int) where {S<:Real}
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 1):lastindex(B, 1), chunksize)
    @batch for r in rs
        _vsample_chunk!(B, A, r)
    end
    return B
end

function _vsample_chunk!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}, ℐ::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    Iₛ, p = A
    K, V = sqhist(p)
    C = vgenerate(K, V, length(ℐ))
    @inbounds for l ∈ eachindex(C, ℐ)
        c = C[l]
        i = ℐ[l]
        B[i, Iₛ[c]] += one(S)
    end
    return B
end
function _vsample_chunk!(B::AbstractMatrix{S}, A::Tuple{AbstractVector{Int}, AbstractVector{T}}, ℐ::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    Iₛ, p = A
    n = length(Iₛ)
    Iₛp = (copyto!(Vector{Int}(undef, n), Iₛ), copyto!(Vector{T}(undef, n), p))
    _vsample_chunk!(B, Iₛp, ℐ)
end

################
# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    C, U = _genstorage_init(Float64, length(ℐ))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            n = length(Iₛ)
            vgenerate!(C, U, n)
            for l ∈ eachindex(C, ℐ)
                c = C[l]
                i = ℐ[l]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of sparse vectors
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {N}
    C, U = _genstorage_init(Float64, length(ℐ))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        n = length(Iₛ)
        vgenerate!(C, U, n)
        for l ∈ eachindex(C, ℐ)
            c = C[l]
            i = ℐ[l]
            B[i, Iₛ[c], IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
_vtsample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
_vtsample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {N} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
function _vtsample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real}
    B = zeros(S, n_sim, n_cat)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix{S}, A::Vector{Int}, chunksize::Int) where {S<:Real}
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 1):lastindex(B, 1), chunksize)
    @batch for r in rs
        _vsample_chunk!(B, A, r)
    end
    return B
end

function _vsample_chunk!(B::AbstractMatrix{S}, A::AbstractVector{Int}, ℐ::UnitRange{Int}) where {S<:Real}
    Iₛ = A
    n = length(Iₛ)
    C = vgenerate(n, length(ℐ))
    @inbounds for l ∈ eachindex(C, ℐ)
        c = C[l]
        i = ℐ[l]
        B[i, Iₛ[c]] += one(S)
    end
    return B
end

################
# General case: dense vectors, the linear index of which indicates the category
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M}
    C, U = _genstorage_init(Float64, length(ℐ))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for p ∈ a
            n = length(p)
            resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
            sqhist!(K, V, large, small, q, p)
            vgenerate!(C, U, K, V)
            for l ∈ eachindex(C, ℐ)
                c = C[l]
                i = ℐ[l]
                B[i, c, IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of dense vectors
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{T}, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    C, U = _genstorage_init(Float64, length(ℐ))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        vgenerate!(C, U, K, V)
        for l ∈ eachindex(C, ℐ)
            c = C[l]
            i = ℐ[l]
            B[i, c, IR] += one(S)
        end
    end
    return B
end

# The simplest case: a dense vector
_vtsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
_vtsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
function _vtsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_sim, n_cat)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix{S}, A::Vector{T}, chunksize::Int) where {S<:Real, T<:AbstractFloat}
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 1):lastindex(B, 1), chunksize)
    @batch for r in rs
        _vsample_chunk!(B, A, r)
    end
    return B
end

function _vsample_chunk!(B::AbstractMatrix{S}, A::AbstractVector{T}, ℐ::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    p = copyto!(Vector{T}(undef, length(A)), A)
    K, V = sqhist(p)
    C = vgenerate(K, V, length(ℐ))
    @inbounds for l ∈ eachindex(C, ℐ)
        c = C[l]
        i = ℐ[l]
        B[i, c] += one(S)
    end
    return B
end

################
# General case: sparse vectors, the nzval of which indicates the category
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M}
    C, U = _genstorage_init(Float64, length(ℐ))
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
            vgenerate!(C, U, K, V)
            for l ∈ eachindex(C, ℐ)
                c = C[l]
                i = ℐ[l]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    return B
end

# A simplification: an array of sparse vectors
function _vsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{SparseVector{Tv, Ti}, N}, keep, default, ℐ::UnitRange{Int}) where {S<:Real, N′} where {Tv<:AbstractFloat, Ti<:Integer, N}
    C, U = _genstorage_init(Float64, length(ℐ))
    K, V, q = _sqhist_init(Tv, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        sv = A[IA]
        Iₛ, p = sv.nzind, sv.nzval
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        vgenerate!(C, U, K, V)
        for l ∈ eachindex(C, ℐ)
            c = C[l]
            i = ℐ[l]
            B[i, Iₛ[c], IR] += one(S)
        end
    end
    return B
end

# The simplest case: a sparse vector
_vtsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::Int, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
_vtsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}, chunksize::Int) where {S<:Real} where {T<:AbstractFloat} where {N} = _vtsample(S, A, n_sim, n_cat, :, chunksize)
function _vtsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, ::Colon, chunksize::Int) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_sim, n_cat)
    vtsample!(B, A, chunksize)
end

function vtsample!(B::AbstractMatrix{S}, A::SparseVector{<:AbstractFloat}, chunksize::Int) where {S<:Real}
    _check_reducedims(B, A)
    rs = splitranges(firstindex(B, 1):lastindex(B, 1), chunksize)
    @batch for r in rs
        _vsample_chunk!(B, A, r)
    end
    return B
end

function _vsample_chunk!(B::AbstractMatrix{S}, A::SparseVector{T}, ℐ::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    Iₛ, p = A.nzind, A.nzval
    K, V = sqhist(p)
    C = vgenerate(K, V, length(ℐ))
    @inbounds for l ∈ eachindex(C, ℐ)
        c = C[l]
        i = ℐ[l]
        B[i, Iₛ[c]] += one(S)
    end
    return B
end
