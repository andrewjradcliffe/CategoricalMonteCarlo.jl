#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

# A ∈ 𝔻ᴰ¹ˣᴰ²ˣᴰ³ˣ⋯ ; eltype(A) = Vector{T} where {T}
#                             T = Vector{Int} if 1. or 2.1.
#                                 NTuple{M, Vector{Int}} where {M} if 2.2.
#                                 (I, 𝐰₂)::Tuple{Vector{Int}, Vector{Float64}} if elaborate

# The abstract PVG algorithm interface may use a composition of types representing
# simple algorithms, ultimately producing a function from a composed type.
#    composed type -> function -> PVG
# Alternatively, one can provide an arbitrary function for the PVG; this enables
# arbitrarily complex algorithms which cannot easily be expressed as some
# composition of simple algorithms. Simple algorithms necessitate a clear flow
# from state to state, whereas in practice, one may wish to re-use a partial
# state from an earlier step, so that a simple composition such as f ∘ g ∘ h
# would fail.

# The result of the function applied to each element of each element of A
# should always be Tuple{Vector{Int}, Vector{<:AbstractFloat}}

# The expected case: eltype(A) as above
function pvg(f::Function, A::AbstractArray{Vector{T}, N}, ws) where {T, N}
    map(a -> map(x -> f(x, ws), a), A)
end

function pvg!(f::Function, B::AbstractArray{Vector{Tuple{Vector{Int}, Vector{S}}}, N}, A::AbstractArray{Vector{T}, N}, ws) where {T, N} where {S<:AbstractFloat}
    for i ∈ eachindex(B, A)
        B[i] = map(x -> f(x, ws), A[i])
    end
    B
end

function pvg(f::Function, A::AbstractArray{Vector{T}, N}, ws::Tuple{}) where {T, N}
    map(a -> map(f, a), A)
end

function pvg!(f::Function, B::AbstractArray{Vector{Tuple{Vector{Int}, Vector{S}}}, N}, A::AbstractArray{Vector{T}, N}, ws::Tuple{}) where {T, N} where {S<:AbstractFloat}
    for i ∈ eachindex(B, A)
        B[i] = map(f, A[i])
    end
    B
end

# A simplification: an array of T, rather than Vector{T}
pvg(f::Function, A::AbstractArray{T, N}, ws) where {T, N} = map(x -> f(x, ws), A)

function pvg!(f::Function, B::AbstractArray{Tuple{Vector{Int}, Vector{S}}, N}, A::AbstractArray{T, N}, ws) where {T, N} where {S<:AbstractFloat}
    for i ∈ eachindex(B, A)
        B[i] = f(A[i], ws)
    end
    B
end

pvg(f::Function, A::AbstractArray{T, N}, ws::Tuple{}) where {T, N} = map(f, A)

function pvg!(f::Function, B::AbstractArray{Tuple{Vector{Int}, Vector{S}}, N}, A::AbstractArray{T, N}, ws::Tuple{}) where {T, N} where {S<:AbstractFloat}
    for i ∈ eachindex(B, A)
        B[i] = f(A[i])
    end
    B
end

# cumulative option: f(I, 𝐰) -> (Iₛ, ω), then g(Iₛ, ω) -> (Iₛ, Σω)
# g(Iₛ, ω) = Iₛ, cumsum(ω) # or, Iₛ, cumsum!(ω)
# g(f, I, 𝐰) = g(f(Iₛ, ω)) # g ∘ f
_g(Iₛ, ω) = Iₛ, cumsum(ω)
_g((Iₛ, ω)) = _g(Iₛ, ω)
# an optimized case for Algorithm1
function _g(Iₛ::Vector{Int})
    N = length(Iₛ)
    c = inv(N)
    Σω = Vector{Float64}(undef, N)
    @inbounds @simd for i ∈ eachindex(Σω)
        Σω[i] = i * c
    end
    Iₛ, Σω
end

pvg_cumulative(f, A, ws) = pvg(_g ∘ f, A, ws)
pvg_cumulative!(f, B, A, ws) = pvg!(_g ∘ f, B, A, ws)
