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
