#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

# # A ∈ 𝔻ᴰ¹ˣᴰ²ˣᴰ³ˣ⋯ ; eltype(A) = Vector{T} where {T}
# #                             T = Vector{Int} if 1. or 2.1.
# #                                 NTuple{M, Vector{Int}} where {M} if 2.2.
# #                                 (I, 𝐰₂)::Tuple{Vector{Int}, Vector{Float64}} if elaborate

# # The abstract PVG algorithm interface may use a composition of types representing
# # simple algorithms, ultimately producing a function from a composed type.
# #    composed type -> function -> PVG
# # Alternatively, one can provide an arbitrary function for the PVG; this enables
# # arbitrarily complex algorithms which cannot easily be expressed as some
# # composition of simple algorithms. Simple algorithms necessitate a clear flow
# # from state to state, whereas in practice, one may wish to re-use a partial
# # state from an earlier step, so that a simple composition such as f ∘ g ∘ h
# # would fail.

# # The input to the function applied to each element of each element of A
# # will generally have a signature which accepts a single argument.
# # The result of the function applied to each element of each element of A
# # should always be Tuple{Vector{Int}, Vector{<:AbstractFloat}}

# # The expected case: eltype(A) as above
# function pvg(f::Function, A::AbstractArray{Vector{T}, N}, ws) where {T, N}
#     map(a -> map(x -> f(x, ws), a), A)
# end

# function pvg!(f::Function, B::AbstractArray{Vector{Tuple{Vector{Int}, Vector{S}}}, N}, A::AbstractArray{Vector{T}, N}, ws) where {T, N} where {S<:AbstractFloat}
#     for i ∈ eachindex(B, A)
#         B[i] = map(x -> f(x, ws), A[i])
#     end
#     B
# end

# function pvg(f::Function, A::AbstractArray{Vector{T}, N}, ws::Tuple{}) where {T, N}
#     map(a -> map(f, a), A)
# end

# function pvg!(f::Function, B::AbstractArray{Vector{Tuple{Vector{Int}, Vector{S}}}, N}, A::AbstractArray{Vector{T}, N}, ws::Tuple{}) where {T, N} where {S<:AbstractFloat}
#     for i ∈ eachindex(B, A)
#         B[i] = map(f, A[i])
#     end
#     B
# end

# # A simplification: an array of T, rather than Vector{T}
# pvg(f::Function, A::AbstractArray{T, N}, ws) where {T, N} = map(x -> f(x, ws), A)

# function pvg!(f::Function, B::AbstractArray{Tuple{Vector{Int}, Vector{S}}, N}, A::AbstractArray{T, N}, ws) where {T, N} where {S<:AbstractFloat}
#     for i ∈ eachindex(B, A)
#         B[i] = f(A[i], ws)
#     end
#     B
# end

# pvg(f::Function, A::AbstractArray{T, N}, ws::Tuple{}) where {T, N} = map(f, A)

# function pvg!(f::Function, B::AbstractArray{Tuple{Vector{Int}, Vector{S}}, N}, A::AbstractArray{T, N}, ws::Tuple{}) where {T, N} where {S<:AbstractFloat}
#     for i ∈ eachindex(B, A)
#         B[i] = f(A[i])
#     end
#     B
# end

# # cumulative option: f(I, 𝐰) -> (Iₛ, ω), then g(Iₛ, ω) -> (Iₛ, Σω)
# # g(Iₛ, ω) = Iₛ, cumsum(ω) # or, Iₛ, cumsum!(ω)
# # g(f, I, 𝐰) = g(f(Iₛ, ω)) # g ∘ f
# _g(Iₛ, ω) = Iₛ, cumsum(ω)
# _g((Iₛ, ω)) = _g(Iₛ, ω)
# # an optimized case for Algorithm_1
# function _g(Iₛ::Vector{Int})
#     N = length(Iₛ)
#     c = inv(N)
#     Σω = Vector{Float64}(undef, N)
#     @inbounds @simd for i ∈ eachindex(Σω)
#         Σω[i] = i * c
#     end
#     Iₛ, Σω
# end

# pvg_cumulative(f, A, ws) = pvg(_g ∘ f, A, ws)
# pvg_cumulative!(f, B, A, ws) = pvg!(_g ∘ f, B, A, ws)

# # Example: elaborate case (I, 𝐰₂)
# #                                  I ∈ ℕᴺ            𝐰₂ ∈ ℝᴺ
# #                                   |                  |
# #                                   v                  v
# # A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{Float64}}}, N} where {N}
# function f(I, 𝐰₂, 𝐰, u)
#     Iₛ = I
#     𝐰₁ = Algorithm_2_1(I, 𝐰)
#     ω₁ = Algorithm_4(𝐰₁, 𝐰₂)
#     ω = Algorithm_3(ω₁, u)
#     Iₛ, ω
# end
# f((I, 𝐰₂), 𝐰, u) = f(I, 𝐰₂, 𝐰, u)
# # closure to provide u; u could also just be hard-coded into original function definition
# f((I, 𝐰₂), 𝐰) = f(I, 𝐰₂, 𝐰, 0.5) # This meets the necessary signature for pvg

# # Example: simple Algorithm 2.2.
# #                                  I₁ ∈ ℕᴺ       I₂ ∈ ℝᴺ
# #                                   |             |
# #                                   v             v
# # A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{Int}}}, N} where {N}
# function f(Is, ws)
#     Iₛ = Is[1]
#     ω = Algorithm_2_2(Is, ws)
#     Iₛ, ω
# end

# # Example: Algorithm 1.
# #                            I₁ ∈ ℕᴺ
# #                             |
# #                             v
# # A::AbstractArray{Vector{Vector{Int}}, N} where {N}
# function f(Iₛ)
#     N = length(Iₛ)
#     ω = fill(inv(N), N)
#     Iₛ, ω
# end
# # This is best handled by a special dispatch as it can be an optimized case. It is
# # also a nice opportunity to be able to just sample such cases.
# # However, if one wants to sample using the in-order traversal, it may be most
# # efficient to use pre-computed Σω's. Thus, generating a ::Tuple{Vector{Int}, Vector{Float64}}
# # for each is a good utility for Algorithm 1. If one wants to sample without allocation,
# # the "default" sample algorithm can be made to dispatch on eltype(A)::Vector{Vector{Int}}
# # Benchmarking is needed to determine if pre-computed Σω's are faster than `rand(Iₛ)`

# # Example: Algorithm 2.1. + Algorithm 3.
# #                            Iₛ ∈ ℕᴺ
# #                             |
# #                             v
# # A::AbstractArray{Vector{Vector{Int}}, N} where {N}
# function f(I, 𝐰, u)
#     Iₛ = I
#     ω₁ = Algorithm_2_1(I, 𝐰)
#     ω = Algorithm_3(ω₁, u)
#     Iₛ, ω
# end
# f(I, 𝐰) = f(I, 𝐰, 0.5) # closure to provide u; or just hard code u

################################################################
# 2022-08-07: revised pvg; useful approach to dealing with type instability
# of what could otherwise be phrased as map(a -> map(x -> (x, f(x)), a), A)
# Needs to be adapted to handle AbstractArray{<:AbstractArray{T}} and also AbstractArray{T}
# ultimately, if `f` is type-stable, then one can just use `map` to create a homogeneously
# typed output; such an approach is more flexible, and defers the details to the user.
_typeoffirstnonempty(f, A) = typeof(f(first(A[findfirst(!isempty, A)])))
function pvg(f, A::AbstractArray{T, N}) where {T<:AbstractArray{S, M}, N} where {S, M}
    Tₒ = _typeoffirstnonempty(f, A)
    B = initialize(Array{Tₒ, M}, size(A))
    pvg!(f, B, A)
end
function pvg!(f, B, A)
    for i ∈ eachindex(B, A)
        !isempty(A[i]) && (B[i] = map(f, A[i]))
    end
    B
end

function tpvg(f, A::AbstractArray{T, N}) where {T<:AbstractArray{S, M}, N} where {S, M}
    Tₒ = _typeoffirstnonempty(f, A)
    B = tinitialize(Array{Tₒ, M}, size(A))
    tpvg!(f, B, A)
end
function tpvg!(f, B, A)
    Threads.@threads for i ∈ eachindex(B, A)
        # !isempty(A[i]) && (B[i] = map(f, A[i]))
        if isempty(A[i])
            empty!(B[i])
        else
            B[i] = map(f, A[i])
        end
    end
    B
end

# Array of array constructors
function initialize!(A::AbstractArray{T, N}) where {T<:AbstractArray, N}
    for i ∈ eachindex(A)
        A[i] = T()
    end
    A
end
function initialize!(A::AbstractArray{T, N}) where {T<:Number, N}
    for i ∈ eachindex(A)
        A[i] = zero(T)
    end
    A
end

_initialize(::Type{T}, dims::NTuple{N, Int}) where {T<:AbstractArray} where {N} = initialize!(Array{T, N}(undef, dims))
_initialize(::Type{T}, dims::NTuple{N, Int}) where {T<:Number} where {N} = zeros(T, dims)
initialize(::Type{T}, dims::NTuple{N, Int}) where {T} where {N} = _initialize(T, dims)
initialize(::Type{T}, dims::Vararg{Integer, N}) where {T} where {N} = initialize(T, (map(Int, dims)...,))

function tinitialize!(A::AbstractArray{T, N}) where {T, N}
    @sync for slc ∈ eachslice(A, dims=N)
        Threads.@spawn initialize!(slc)
    end
    A
end
tinitialize(::Type{T}, dims::NTuple{N, Int}) where {T<:AbstractArray} where {N} = tinitialize!(Array{T, N}(undef, dims))
tinitialize(::Type{T}, dims::Vararg{Integer, N}) where {T} where {N} = tinitialize(T, (map(Int, dims)...,))
