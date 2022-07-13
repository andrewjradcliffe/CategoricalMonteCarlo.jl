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

vsample(::Type{S}, A, n_sim, n_cat; dims=:) where {S} = vsample(S, A, n_sim, n_cat, dims)
vsample(::Type{S}, A, n_sim; dims=:) where {S} = vsample(S, A, n_sim, num_cat(A), dims)
vsample(::Type{S}, A, n_sim::Int, n_cat::Int, dims::Int) where {S} = vsample(S, A, n_sim, n_cat, (dims,))

function vsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    vsample!(B, A)
end

function vsample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T, N}
    B = fill!(similar(A, S, (n_cat, n_sim)), zero(S))
    vsample!(B, A)
end

# The expected case: vectors of sparse vectors (as their bare components)
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
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
            for (j′, j) ∈ enumerate(axes(B, 2))
                c = C[j′]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        vgenerate!(C, U, K, V)
        for (j′, j) ∈ enumerate(axes(B, 2))
            c = C[j′]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

# The simplest case: a sparse vector
vsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = vsample(S, A, n_sim, n_cat, :)
vsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = vsample(S, A, n_sim, n_cat, :)

function vsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat} where {N}
    B = zeros(S, n_cat, n_sim)
    vsample!(B, A)
end

function vsample!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    Iₛ, p = A
    K, V = sqhist(p)
    C = vgenerate(K, V, size(B, 2))
    @inbounds for (j′, j) ∈ enumerate(axes(B, 2))
        c = C[j′]
        B[Iₛ[c], j] += one(S)
    end
    B
end

################
# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            n = length(Iₛ)
            vgenerate!(C, U, n)
            for (j′, j) ∈ enumerate(axes(B, 2))
                c = C[j′]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        n = length(Iₛ)
        vgenerate!(C, U, n)
        for (j′, j) ∈ enumerate(axes(B, 2))
            c = C[j′]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

# The simplest case: a sparse vector
function vsample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {N}
    B = zeros(S, n_cat, n_sim)
    vsample!(B, A)
end

# Oddly, the fastest vsampler is non-allocating -- most likely due to
# the elimination of store + access instructions associated with using a temporary array.
function vsample!(B::AbstractMatrix{S}, Iₛ::Vector{Int}) where {S<:Real}
    _check_reducedims(B, Iₛ)
    n = length(Iₛ)
    C = vgenerate(n, size(B, 2))
    @inbounds for (j′, j) ∈ enumerate(axes(B, 2))
        c = C[j′]
        B[Iₛ[c], j] += one(S)
    end
    B
end

################
# General case: dense vectors, the linear index of which indicates the category
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
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
            for (j′, j) ∈ enumerate(axes(B, 2))
                c = C[j′]
                B[c, j, IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of dense vectors
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{T}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        vgenerate!(C, U, K, V)
        for (j′, j) ∈ enumerate(axes(B, 2))
            c = C[j′]
            B[c, j, IR] += one(S)
        end
    end
    B
end

# The simplest case: a dense vector
vsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = vsample(S, A, n_sim, n_cat, :)
vsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = vsample(S, A, n_sim, n_cat, :)

function vsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat} where {N}
    B = zeros(S, n_cat, n_sim)
    vsample!(B, A)
end

function vsample!(B::AbstractMatrix{S}, A::Vector{T}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    K, V = sqhist(A)
    C = vgenerate(K, V, size(B, 2))
    @inbounds for (j′, j) ∈ enumerate(axes(B, 2))
        c = C[j′]
        B[c, j] += one(S)
    end
    B
end


################
# General case: sparse vectors, the nzval of which indicates the category
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
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
            for (j′, j) ∈ enumerate(axes(B, 2))
                c = C[j′]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function vsample!(B::AbstractArray{S, N′}, A::AbstractArray{SparseVector{Tv, Ti}, N}) where {S<:Real, N′} where {Tv<:AbstractFloat, Ti<:Integer, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 2))
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
        for (j′, j) ∈ enumerate(axes(B, 2))
            c = C[j′]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

# The simplest case: a sparse vector
vsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = vsample(S, A, n_sim, n_cat, :)
vsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = vsample(S, A, n_sim, n_cat, :)

function vsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat} where {N}
    B = zeros(S, n_cat, n_sim)
    vsample!(B, A)
end

function vsample!(B::AbstractMatrix{S}, A::SparseVector{T}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    (; n, nzind, nzval) = A
    Iₛ, p = A.nzind, A.nzval
    K, V = sqhist(p)
    C = vgenerate(K, V, size(B, 2))
    @inbounds for (j′, j) ∈ enumerate(axes(B, 2))
        c = C[j′]
        B[Iₛ[c], j] += one(S)
    end
    B
end
