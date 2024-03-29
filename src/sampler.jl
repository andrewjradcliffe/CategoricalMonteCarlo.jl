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

# sample(::Type{S}, A, n_sim, n_cat; dims=:) where {S} = sample(S, A, n_sim, n_cat, dims)
# sample(::Type{S}, A, n_sim; dims=:) where {S} = sample(S, A, n_sim, num_cat(A), dims)
# sample(::Type{S}, A, n_sim::Int, n_cat::Int, dims::Int) where {S} = sample(S, A, n_sim, n_cat, (dims,))

# function sample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T, N}
#     Dᴬ = size(A)
#     Dᴮ = tuple(n_sim, n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
#     B = fill!(similar(A, S, Dᴮ), zero(S))
#     sample!(B, A)
# end

# function sample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T, N}
#     B = fill!(similar(A, S, (n_sim, n_cat)), zero(S))
#     sample!(B, A)
# end

#### A revised public interface
"""
    sample(::Type{T<:Real}=Int, A::AbstractArray, n_sim::Int; [dims=:], [n_cat=nothing])

Draw `n_sim` samples from the distribution which corresponds to the sum of the independent
categorical distributions defined by the probability mass vector(s), `A`, storing the result
in an array of `eltype` `T` of sufficient size and dimension as determined by the input, `A`,
the number of categories, `n_cat`, and the (potential) reduction dimensions, `dims`.

`dims` is an optional keyword argument used to specify an in-place sum on the indices of `A`
(if `A` is an array of arrays). `n_cat` may be optionally specified, or inferred from the
available data; validity indices will be checked regardless.

In the simplest case, `A` may be a single probability mass vector, for which compatible types
are `AbstractVector{<:Real}`, `SparseVector{<:Real, <:Integer}`,
and `AbstractVector{Tuple{<:Integer, <:Real}}`, the last of which is simply another
representation of a sparse vector.

In the second case, `A` may be an `AbstractArray{V, N} where {V, N}` in which `V` is any
of the types specified above for a single probability mass vector. That is, an array
of vectors.

In the third case, `A` may be an `AbstractArray{W, M} where {W<:AbstractArray{V, N}, M} where {V, N}`
in which `W` is any of the types specified in the second case. In other words, an
array of arrays of vectors.

See also: [`tsample`](@ref), [`vsample`](@ref), [`vtsample`](@ref)
"""
sample(::Type{S}, A, n_sim; dims=:, n_cat=nothing) where {S<:Real} = _sample(S, A, n_sim, n_cat, dims)
sample(A, n_sim; dims=:, n_cat=nothing) = _sample(Int, A, n_sim, n_cat, dims)

_sample(::Type{S}, A, n_sim, n_cat::Int, dims::Int) where {S<:Real} = _sample(S, A, n_sim, n_cat, (dims,))
_sample(::Type{S}, A, n_sim, ::Nothing, dims) where {S<:Real} = _sample(S, A, n_sim, num_cat(A), dims)

function _sample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_sim, n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = fill!(similar(A, S, Dᴮ), zero(S))
    sample!(B, A)
end

function _sample(::Type{S}, A::AbstractArray{T, N}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T, N}
    B = fill!(similar(A, S, (n_sim, n_cat)), zero(S))
    sample!(B, A)
end

################
# The expected case: vectors of sparse vectors (as their bare components)
"""
    sample!(B::AbstractArray, A::AbstractArray)

Draw samples, summing (and potentially reducing) in-place into `B`. The shape of `B`
determines the extent of reduction performed.

See also: [`tsample!`](@ref), [`vsample!`](@ref), [`vtsample!`](@ref)
"""
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
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
            for (i′, i) ∈ enumerate(axes(B, 1))
                c = C[i′]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors (as bare components)
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        generate!(C, U, K, V)
        for (i′, i) ∈ enumerate(axes(B, 1))
            c = C[i′]
            B[i, Iₛ[c], IR] += one(S)
        end
    end
    B
end

# The simplest case: a sparse vector (as bare components)
_sample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = _sample(S, A, n_sim, n_cat, :)
_sample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = _sample(S, A, n_sim, n_cat, :)
function _sample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_sim, n_cat)
    sample!(B, A)
end

function sample!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    Iₛ, p = A
    K, V = sqhist(p)
    C = generate(K, V, size(B, 1))
    @inbounds for (i′, i) ∈ enumerate(axes(B, 1))
        c = C[i′]
        B[i, Iₛ[c]] += one(S)
    end
    B
end

################
# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            n = length(Iₛ)
            generate!(C, U, n)
            for (i′, i) ∈ enumerate(axes(B, 1))
                c = C[i′]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ = A[IA]
        n = length(Iₛ)
        generate!(C, U, n)
        for (i′, i) ∈ enumerate(axes(B, 1))
            c = C[i′]
            B[i, Iₛ[c], IR] += one(S)
        end
    end
    B
end

# # The simplest case: a sparse vector
_sample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} = _sample(S, A, n_sim, n_cat, :)
_sample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {N} = _sample(S, A, n_sim, n_cat, :)
function _sample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real}
    B = zeros(S, n_sim, n_cat)
    sample!(B, A)
end

# Oddly, the fastest sampler is non-allocating -- most likely due to
# the elimination of store + access instructions associated with using a temporary array.
function sample!(B::AbstractMatrix{S}, Iₛ::Vector{Int}) where {S<:Real}
    _check_reducedims(B, Iₛ)
    n = length(Iₛ)
    C = generate(n, size(B, 1))
    @inbounds for (i′, i) ∈ enumerate(axes(B, 1))
        c = C[i′]
        B[i, Iₛ[c]] += one(S)
    end
    B
end

################
# General case: dense vectors, the linear index of which indicates the category
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
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
            for (i′, i) ∈ enumerate(axes(B, 1))
                c = C[i′]
                B[i, c, IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of dense vectors
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{T}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
    K, V, q = _sqhist_init(T, 0)
    large, small = _largesmall_init(0)
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        p = A[IA]
        n = length(p)
        resize!(K, n); resize!(V, n); resize!(large, n); resize!(small, n); resize!(q, n)
        sqhist!(K, V, large, small, q, p)
        generate!(C, U, K, V)
        for (i′, i) ∈ enumerate(axes(B, 1))
            c = C[i′]
            B[i, c, IR] += one(S)
        end
    end
    B
end

# The simplest case: a dense vector
_sample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = _sample(S, A, n_sim, n_cat, :)
_sample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = _sample(S, A, n_sim, n_cat, :)
function _sample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_sim, n_cat)
    sample!(B, A)
end

function sample!(B::AbstractMatrix{S}, A::Vector{T}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    K, V = sqhist(A)
    C = generate(K, V, size(B, 1))
    @inbounds for (i′, i) ∈ enumerate(axes(B, 1))
        c = C[i′]
        B[i, c] += one(S)
    end
    B
end


################
# General case: sparse vectors, the nzval of which indicates the category
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
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
            for (i′, i) ∈ enumerate(axes(B, 1))
                c = C[i′]
                B[i, Iₛ[c], IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function sample!(B::AbstractArray{S, N′}, A::AbstractArray{SparseVector{Tv, Ti}, N}) where {S<:Real, N′} where {Tv<:AbstractFloat, Ti<:Integer, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C, U = _genstorage_init(Float64, size(B, 1))
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
        for (i′, i) ∈ enumerate(axes(B, 1))
            c = C[i′]
            B[i, Iₛ[c], IR] += one(S)
        end
    end
    B
end

# The simplest case: a sparse vector
_sample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = _sample(S, A, n_sim, n_cat, :)
_sample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = _sample(S, A, n_sim, n_cat, :)

function _sample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_sim, n_cat)
    sample!(B, A)
end

function sample!(B::AbstractMatrix{S}, A::SparseVector{T}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    Iₛ, p = A.nzind, A.nzval
    K, V = sqhist(p)
    C = generate(K, V, size(B, 1))
    @inbounds for (i′, i) ∈ enumerate(axes(B, 1))
        c = C[i′]
        B[i, Iₛ[c]] += one(S)
    end
    B
end
