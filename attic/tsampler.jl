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

# The expected case: vectors of sparse vectors (as their bare components)
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C, U = _genstorage_init(Float64, L)
        K, V, ix, q = _marsaglia_init(T)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for (Iₛ, ω) ∈ a
                n = length(ω)
                resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
                marsaglia!(K, V, q, ix, ω)
                marsaglia_generate!(C, U, K, V)
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

# A simplification: an array of sparse vectors
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C, U = _genstorage_init(Float64, L)
        K, V, ix, q = _marsaglia_init(T)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            Iₛ, ω = A[IA]
            n = length(ω)
            resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
            marsaglia!(K, V, q, ix, ω)
            marsaglia_generate!(C, U, K, V)
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

# The simplest case: a sparse vector
tsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = tsample(S, A, n_sim, n_cat, :)
tsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = tsample(S, A, n_sim, n_cat, :)

function tsample(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A)
end

function tsample!(B::AbstractMatrix, A::Tuple{Vector{Int}, Vector{<:AbstractFloat}})
    _check_reducedims(B, A)
    tsample!(B, A, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1048576
        Iₛ, ω = A
        K, V = marsaglia(ω)
        n = length(K)
        @inbounds for j ∈ 𝒥
            u = rand()
            j′ = floor(Int, muladd(u, n, 1))
            c = u < V[j′] ? j′ : K[j′]
            B[Iₛ[c], j] += one(S)
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample!(B, A, start:h)
            tsample!(B, A, (h + 1):stop)
        end
        return B
    end
end

################
# Specialized method for eltype(A)::Vector{Vector{Int}}
# or, in other words, where the probability mass on each element is 1 / length(Iₛ)
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
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
                marsaglia_generate!(C, U, K, V)
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

# A simplification: an array of sparse vectors
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Int}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
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
            marsaglia_generate!(C, U, K, V)
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

# The simplest case: a sparse vector
function tsample(::Type{S}, A::Vector{Int}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {N}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A)
end

# Trivial parallelism is preferable here, but it's not safe!
# These are questionable methods (though, the function barrier approach is safe).
# @inline function _tsample!(B::AbstractMatrix{S}, A::Vector{Int}, j::Int) where {S<:Real}
#     c = rand(A)
#     @inbounds B[c, j] += one(S)
#     B
# end
# function tsample0!(B::AbstractMatrix{S}, A::Vector{Int}) where {S<:Real}
#     _check_reducedims(B, A)
#     # @inbounds Threads.@threads for j ∈ axes(B, 2)
#     #     c = rand(A)
#     #     B[c, j] += one(S)
#     # end
#     @inbounds Threads.@threads for j ∈ axes(B, 2)
#         _tsample!(B, A, j)
#     end
#     B
# end

function tsample!(B::AbstractMatrix, A::Vector{Int})
    _check_reducedims(B, A)
    tsample!(B, A, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractMatrix{S}, A::Vector{Int}, 𝒥::UnitRange{Int}) where {S<:Real}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1048576
        n = length(A)
        K, V = marsaglia(fill(inv(n), n))
        @inbounds for j ∈ 𝒥
            u = rand()
            j′ = floor(Int, muladd(u, n, 1))
            c = u < V[j′] ? j′ : K[j′]
            B[A[c], j] += one(S)
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample!(B, A, start:h)
            tsample!(B, A, (h + 1):stop)
        end
        return B
    end
end


################
# General case: dense vectors, the linear index of which indicates the category
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{T}, M}, N} where {T<:AbstractFloat, M}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C, U = _genstorage_init(Float64, L)
        K, V, ix, q = _marsaglia_init(T)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for ω ∈ a
                n = length(ω)
                resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
                marsaglia!(K, V, q, ix, ω)
                marsaglia_generate!(C, U, K, V)
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

# A simplification: an array of dense vectors
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{T}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C, U = _genstorage_init(Float64, L)
        K, V, ix, q = _marsaglia_init(T)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            ω = A[IA]
            n = length(ω)
            resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
            marsaglia!(K, V, q, ix, ω)
            marsaglia_generate!(C, U, K, V)
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

# The simplest case: a dense vector
tsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = tsample(S, A, n_sim, n_cat, :)
tsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = tsample(S, A, n_sim, n_cat, :)

function tsample(::Type{S}, A::Vector{T}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A)
end

function tsample!(B::AbstractMatrix, A::Vector{<:AbstractFloat})
    _check_reducedims(B, A)
    tsample!(B, A, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractMatrix{S}, A::Vector{T}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1048576
        ω = A
        K, V = marsaglia(ω)
        n = length(K)
        @inbounds for j ∈ 𝒥
            u = rand()
            j′ = floor(Int, muladd(u, n, 1))
            c = u < V[j′] ? j′ : K[j′]
            B[c, j] += one(S)
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample!(B, A, start:h)
            tsample!(B, A, (h + 1):stop)
        end
        return B
    end
end


################
# General case: sparse vectors, the nzval of which indicates the category
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{SparseVector{Tv, Ti}, M}, N} where {Tv<:AbstractFloat, Ti<:Integer, M}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
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
                marsaglia_generate!(C, U, K, V)
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

# A simplification: an array of sparse vectors
function tsample!(B::AbstractArray{S, N′}, A::AbstractArray{SparseVector{Tv, Ti}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {Tv<:AbstractFloat, Ti<:Integer, N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
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
            marsaglia_generate!(C, U, K, V)
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

# The simplest case: a sparse vector
tsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = tsample(S, A, n_sim, n_cat, :)
tsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = tsample(S, A, n_sim, n_cat, :)

function tsample(::Type{S}, A::SparseVector{T}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat}
    B = zeros(S, n_cat, n_sim)
    tsample!(B, A)
end

function tsample!(B::AbstractMatrix, A::SparseVector{<:AbstractFloat})
    _check_reducedims(B, A)
    tsample!(B, A, firstindex(B, 2):size(B, 2))
end

function tsample!(B::AbstractMatrix{S}, A::SparseVector{T}, 𝒥::UnitRange{Int}) where {S<:Real} where {T<:AbstractFloat}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1048576
        (; n, nzind, nzval) = A
        Iₛ, ω = nzind, nzval
        K, V = marsaglia(ω)
        n = length(K)
        @inbounds for j ∈ 𝒥
            u = rand()
            j′ = floor(Int, muladd(u, n, 1))
            c = u < V[j′] ? j′ : K[j′]
            B[Iₛ[c], j] += one(S)
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample!(B, A, start:h)
            tsample!(B, A, (h + 1):stop)
        end
        return B
    end
end
