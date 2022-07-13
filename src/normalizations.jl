#
# Date created: 2022-06-15
# Author: aradclif
#
#
############################################################################################

_typeofinv(x) = typeof(inv(x))
_typeofinv(::Type{T}) where {T} = typeof(inv(one(T)))

"""
    normalize1!(A::AbstractArray{<:Real})

Normalize the values in `A` such that `sum(A) ≈ 1` and `0 ≤ A[i] ≤ 1` ∀i.
This is not quite the L¹-norm, which would require that `abs(A[i])` be used.
It is assumed that `0 ≤ A[i] < Inf` ∀i. `Inf` values are not handled and
will result in `NaN`'s.

See also: [`normalize1`](@ref)

```jldoctest
julia> normalize1!([1.0, 2.0, 3.0])
3-element Vector{Float64}:
 0.16666666666666666
 0.3333333333333333
 0.5

julia> normalize1!([1.0, 2.0, Inf])
3-element Vector{Float64}:
   0.0
   0.0
 NaN

julia> normalize1!([1.0, 2.0, NaN])     # NaN propagates, as expected
3-element Vector{Float64}:
 NaN
 NaN
 NaN

julia> normalize1!([1.0, -2.0, 3.0])    # not the L¹-norm
3-element Vector{Float64}:
  0.5
 -1.0
  1.5
```
"""
function normalize1!(A::AbstractArray{T}) where {T<:Real}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @inbounds @simd for i ∈ eachindex(A)
        A[i] *= c
    end
    A
end

"""
    normalize1!(B::AbstractArray{<:Real}, A::AbstractArray{<:Real})

Normalize the values in `A` such that `sum(B) ≈ 1` and `0 ≤ B[i] ≤ 1` ∀i, storing
the result in `B`. It is assumed that `A[i] ≥ 0` ∀i.
"""
function normalize1!(B::AbstractArray{T}, A::AbstractArray{S}) where {T<:Real, S<:Real}
    s = zero(S)
    @inbounds @simd for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @inbounds @simd for i ∈ eachindex(A, B)
        B[i] = A[i] * c
    end
    B
end

"""
    normalize1(A::AbstractArray{<:Real})

Return an array of equal size which satisfies `sum(B) ≈ 1` and `0 ≤ B[i] 1` ∀i.
It is assumed that `A[i] ≥ 0` ∀i.

See also: [`normalize1!`](@ref)
"""
normalize1(A::AbstractArray{T}) where {T<:Real} = normalize1!(similar(A, _typeofinv(T)), A)

function vnormalize1!(A::AbstractArray{T}) where {T<:Base.IEEEFloat}
    s = zero(T)
    @turbo for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @turbo for i ∈ eachindex(A)
        A[i] *= c
    end
    A
end

function normalize1!(B::AbstractArray{T}, A::AbstractArray{T}) where {T<:Base.IEEEFloat}
    s = zero(S)
    @turbo for i ∈ eachindex(A)
        s += A[i]
    end
    c = inv(s)
    @turbo for i ∈ eachindex(A, B)
        B[i] = A[i] * c
    end
    B
end

normalize1(A::AbstractArray{<:Base.IEEEFloat}) = normalize1!(similar(A), A)

################
# @noinline function _check_algorithm2_1(I::Vector{Int}, x)
#     mn, mx = extrema(I)
#     f, l = firstindex(x), lastindex(x)
#     mn ≥ f || throw(BoundsError(x, mn))
#     mx ≤ l || throw(BoundsError(x, mx))
# end

#### Algorithm 2.1.
# I ∈ ℕᴺ, 𝐰 ∈ ℝᴰ; I ⊆ 1,…,D
# -> ω ∈ ℝᴺ, ωᵢ = 𝐰ᵢ / ∑ⱼ 𝐰ⱼ; j ∈ I

function unsafe_algorithm2_1!(p::Vector{T}, I::Vector{Int}, w::Vector{S}) where {T<:Real, S<:Real}
    s = zero(S)
    @inbounds @simd ivdep for i ∈ eachindex(I, p)
        w̃ = w[I[i]]
        s += w̃
        p[i] = w̃
    end
    # c = inv(s)
    # Guarantees type stability at negligible expense compared to what is gained
    c = one(T) / s
    @inbounds @simd for i ∈ eachindex(p)
        p[i] *= c
    end
    return p
end

"""
    algorithm2_1!(p::Vector{T}, I::Vector{Int}, w::Vector{S}) where {T<:Real, S<:Real}

Fill `p` with the probabilities that result from normalizing the weights selected by `I` from `w`.
Note that `T` must be a type which is able to hold the result of `inv(one(S))`.

See also: [`algorithm2_1`](@ref)
"""
function algorithm2_1!(p::Vector{T}, I::Vector{Int}, w::Vector{S}) where {T<:Real, S<:Real}
    checkbounds(w, I)
    unsafe_algorithm2_1!(p, I, w)
end

"""
    algorithm2_1(I::Vector{Int}, w::Vector{<:Real})

Create a vector of probabilities by normalizing the weights selected by `I` from `w`.
It is assumed that `0 ≤ wᵢ < Inf` and that `NaN`'s are not present, at least for
the (sub)set `w[I]`.

Mathematically, given:

I ∈ ℕᴺ, 𝐰 ∈ ℝᴰ; I ⊆ {1,…,D}

The iᵗʰ term will be computed as: pᵢ = 𝐰ᵢ / ∑ⱼ 𝐰ⱼ; j ∈ I

See also: [`algorithm2_1!`](@ref)

# Examples
```jldoctest
julia> I = [1, 5, 2]; w = [5, 4, 3, 2, 1];

julia> algorithm2_1(I, w)
3-element Vector{Float64}:
 0.5
 0.1
 0.4

julia> algorithm2_1(I, Rational.(w))
3-element Vector{Rational{Int64}}:
 1//2
 1//10
 2//5

julia> w[2] = -w[2];

julia> algorithm2_1(I, w)                # Nonsense results if `wᵢ` constraints violated
3-element Vector{Float64}:
  2.5
  0.5
 -2.0

julia> algorithm2_1(I, [5, 4, 3, 2, Inf])
3-element Vector{Float64}:
   0.0
 NaN
   0.0
```
"""
algorithm2_1(I::Vector{Int}, w::Vector{T}) where {T<:Real} = algorithm2_1!(similar(I, _typeofinv(T)), I, w)

#### Algorithm 2.2
# I₁ ∈ ℕᴺ, I₂ ∈ ℕᴺ, …, Iₘ ∈ ℕᴺ; 𝐰₁ ∈ ℝᴰ¹, 𝐰₂ ∈ ℝᴰ², …, 𝐰ₘ ∈ ℝᴰᵐ
# -> ω ∈ ℝᴺ, ωᵢ = ∏ₘ₌₁ᴹ 𝐰ₘ[Iₘ[i]] / ∑ᵢ₌₁ᴺ ∏ₘ₌₁ᴹ 𝐰ₘ[Iₘ[i]]

function algorithm2_2_quote(M::Int)
    Is = Expr(:tuple)
    ws = Expr(:tuple)
    bc = Expr(:tuple)
    for m = 1:M
        push!(Is.args, Symbol(:I_, m))
        push!(ws.args, Symbol(:w_, m))
        push!(bc.args, Expr(:call, :checkbounds, Symbol(:w_, m), Symbol(:I_, m)))
    end
    block = Expr(:block)
    loop = Expr(:for, Expr(:(=), :j, Expr(:call, :eachindex, ntuple(i -> Symbol(:I_, i), M)..., :p)), block)
    e = Expr(:call, :*)
    for m = 1:M
        push!(e.args, Expr(:ref, Symbol(:w_, m), Expr(:ref, Symbol(:I_, m), :j)))
    end
    push!(block.args, Expr(:(=), :t, e))
    push!(block.args, Expr(:(=), :s, Expr(:call, :+, :s, :t)))
    push!(block.args, Expr(:(=), Expr(:ref, :p, :j), :t))
    return quote
        $Is = Is
        $ws = ws
        $bc
        s = zero(T)
        @inbounds @simd ivdep $loop
        c = inv(s)
        @inbounds @simd ivdep for j ∈ eachindex(p)
            p[j] *= c
        end
        return p
    end
end
@generated function algorithm2_2!(p, Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real}
    algorithm2_2_quote(M)
end

"""
    algorithm2_2!(p, Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real}

Compute the product of weights selected by the respective index sets `Is`,
then normalize the resultant weight vector to probabilities, storing the result in `p`.

See also: [`algorithm2_2`](@ref)

# Examples
```jldoctest
julia> Is = ([1, 2, 3], [4, 5, 6], [7, 8, 9]); ws = ([1.0, 2.0, 3.0], fill(0.5, 6), fill(0.1, 9));

julia> algorithm2_2!(zeros(3), Is, ws)
3-element Vector{Float64}:
 0.16666666666666666
 0.3333333333333333
 0.5
```
"""
algorithm2_2!(p, Is::NTuple{1, Vector{Int}}, ws::NTuple{1, Vector{T}}) where {M} where {T<:Real} = algorithm2_1!(p, (@inbounds Is[1]), (@inbounds ws[1]))

"""
    algorithm2_2(Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real}

Compute the product of weights selected by the respective index sets `Is`,
then normalize the resultant weight vector to probabilities.
Mathematically, given:

I₁ ∈ ℕᴺ , 𝐰₁ ∈ ℝᴰ¹

I₂ ∈ ℕᴺ , 𝐰₂ ∈ ℝᴰ²

⋮       , ⋮

Iₘ ∈ ℕᴺ , 𝐰ₘ ∈ ℝᴰᵐ

The iᵗʰ term will be computed as:
pᵢ = ∏ₘ₌₁ᴹ 𝐰ₘ[Iₘ[i]] / ∑ⱼ₌₁ᴺ ∏ₘ₌₁ᴹ 𝐰ₘ[Iₘ[j]]

See also: [`algorithm2_2!`](@ref)

# Examples
```jldoctest
julia> Is = ([1, 2, 3], [4, 5, 6], [7, 8, 9]); ws = ([1.0, 2.0, 3.0], fill(0.5, 6), fill(0.1, 9));

julia> algorithm2_2(Is, ws)
3-element Vector{Float64}:
 0.16666666666666666
 0.3333333333333333
 0.5

julia> w = ws[1][Is[1]] .* ws[2][Is[2]] .* ws[3][Is[3]]    # unnormalized
3-element Vector{Float64}:
 0.05
 0.1
 0.15000000000000002

julia> w ./= sum(w)                                        # normalized
3-element Vector{Float64}:
 0.16666666666666666
 0.3333333333333333
 0.5
```
"""
algorithm2_2(Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real} = algorithm2_2!(Vector{Base.promote_op(inv, T)}(undef, maximum(length, Is)), Is, ws)

# function algorithm2_2_weightonly_quote(M::Int)
#     Is = Expr(:tuple)
#     ws = Expr(:tuple)
#     bc = Expr(:tuple)
#     for m = 1:M
#         push!(Is.args, Symbol(:I_, m))
#         push!(ws.args, Symbol(:w_, m))
#         push!(bc.args, Expr(:call, :checkbounds, Symbol(:w_, m), Symbol(:I_, m)))
#     end
#     block = Expr(:block)
#     loop = Expr(:for, Expr(:(=), :j, Expr(:call, :eachindex, ntuple(i -> Symbol(:I_, i), M)..., :w′)), block)
#     e = Expr(:call, :*)
#     for m = 1:M
#         push!(e.args, Expr(:ref, Symbol(:w_, m), Expr(:ref, Symbol(:I_, m), :j)))
#     end
#     push!(block.args, Expr(:(=), Expr(:ref, :w′, :j), e))
#     return quote
#         $Is = Is
#         $ws = ws
#         $bc
#         @inbounds @simd ivdep $loop
#         return w′
#     end
# end

# """
#     algorithm2_2_weightonly!(w, Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real}

# Compute the local product of weights, storing the result in `w`.

# See also: [`algorithm2_2_weightonly`](@ref)
# """
# @generated function algorithm2_2_weightonly!(w′, Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real}
#     algorithm2_2_weightonly_quote(M)
# end

# """
#     algorithm2_2_weightonly(Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real}

# Compute the local product of weights identified by the index sets `Is`, which select
# the desired terms from the global weights `ws`. Mathematically, given:

# I₁ ∈ ℕᴺ , 𝐰₁ ∈ ℝᴰ¹

# I₂ ∈ ℕᴺ , 𝐰₂ ∈ ℝᴰ²

# ⋮       , ⋮

# Iₘ ∈ ℕᴺ , 𝐰ₘ ∈ ℝᴰᵐ

# The iᵗʰ term will be computed as:
# wᵢ′ = ∏ₘ₌₁ᴹ 𝐰ₘ[Iₘ[i]] = ∏ₘ₌₁ᴹ 𝐰ₘ,ⱼ : j = Iₘ[i]

# See also: [`algorithm2_2_weightonly!`](@ref)
# """
# algorithm2_2_weightonly(Is::NTuple{M, Vector{Int}}, ws::NTuple{M, Vector{T}}) where {M} where {T<:Real} =
#     algorithm2_2_weightonly!(Vector{T}(undef, maximum(length, Is)), Is, ws)


################

#### Algorithm 3. -- FillMass
# 𝐰 ∈ ℝᴺ, u ∈ ℝ, 0 ≤ u ≤ 1
# -> ω ∈ ℝᴺ, J = {i: wᵢ = 0}
# ωᵢ =
#     Case 1: if J ≠ ∅
#             u / |J|                     if i ∈ J
#             (1 - u) * 𝐰ᵢ / ∑ᵢ₌₁ᴺ 𝐰ᵢ     otherwise
#     Case 2: if J = 1,…,N
#             1/N

## Alternative using ratio
# r ∈ ℝ
# r = u / (1 - u)    ⟹    u = r / (1 + r)
# _r(u::T) where {T<:Real} = u / (one(T) - u)
_u(r::T) where {T<:Real} = r / (one(T) + r)

_check_u01(u::S) where {S<:Real} = (zero(S) ≤ u ≤ one(S) || throw(DomainError(u, "u must be: $(zero(S)) ≤ u ≤ $(one(S))")))

"""
    algorithm3!(p::Vector{T}, u::Real) where {T<:Real}

Normalize `p` to probabilities, spreading probability mass `u` across the
0 or more elements of `p` which are equal to zero. If all values of `p` are equal
to zero, `p` is filled with `1 / length(p)`.
Refer to the respective documentation for a description of `algorithm3`.
Note that `T` must be a type which is able to hold the result of `inv(one(T))`.

See also: [`algorithm3`](@ref), [`algorithm3_ratio!`](@ref)

# Examples
```
julia> algorithm3!(Rational{Int}[0, 10, 5, 0], 0.5)
4-element Vector{Rational{Int64}}:
 1//4
 1//3
 1//6
 1//4
```
"""
function algorithm3!(p::Vector{T}, u::T) where {T<:Real}
    _check_u01(u)
    s = zero(T)
    z = 0
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        s += pᵢ
        z += pᵢ == zero(T)
    end
    c = z == 0 ? inv(s) : (one(T) - u) / s
    u′ = z == length(p) ? one(T) / z : u / z
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = pᵢ == zero(T) ? u′ : pᵢ * c
    end
    p
end
algorithm3!(p::Vector{T}, u::S) where {T<:Real, S<:Real} = algorithm3!(p, convert(T, u))

"""
    algorithm3!(p::Vector{T}, w::Vector{<:Real}, u::Real) where {T<:Real}

Normalize `w` to probabilities, storing the result in `p`, spreading probability
mass `0 ≤ u ≤ 1` across the 0 or more elements of `w` which are equal to zero.
If all values of `w` are zero, `p` is filled with `1 / length(p)`.
Note that `T` must be a type which is able to hold the result of `inv(one(T))`.
"""
function algorithm3!(p::Vector{S}, w::Vector{T}, u::S) where {S<:Real, T<:Real}
    _check_u01(u)
    s = zero(T)
    z = 0
    @inbounds @simd for i ∈ eachindex(p, w)
        w̃ = w[i]
        s += w̃
        p[i] = w̃
        z += w̃ == zero(T)
    end
    c = z == 0 ? one(S) / s : (one(S) - u) / s
    u′ = z == length(p) ? one(S) / z : u / z
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = pᵢ == zero(S) ? u′ : pᵢ * c
    end
    p
end
algorithm3!(p::Vector{S}, w::Vector{T}, u::U) where {S<:Real, T<:Real, U<:Real} = algorithm3!(p, w, convert(S, u))

"""
    algorithm3(w::Vector{<:Real}, u::Real)

Return a vector of probabilities created by normalizing `w` to probabilities, then
spreading the probability mass `0 ≤ u ≤ 1` across the 0 or more elements of `w` which
are equal to zero. If all values of `w` are zero, `p` is filled with `1 / length(p)`.

Mathematically, given:

𝐰 ∈ ℝᴺ, u ∈ ℝ, 0 ≤ u ≤ 1, J = {i : 𝐰ᵢ = 0}

```
pᵢ =
    Case 1: if J ≠ ∅
            u / |J|                     if i ∈ J
            (1 - u) * 𝐰ᵢ / ∑ᵢ₌₁ᴺ 𝐰ᵢ     otherwise
    Case 2: if J = {1,…,N}
            1/N
```

See also: [`algorithm3!`](@ref), [`algorithm3_ratio`](@ref)

# Examples
```jldoctest
julia> algorithm3([0, 10, 5, 0], 0.5)
4-element Vector{Float64}:
 0.25
 0.3333333333333333
 0.16666666666666666
 0.25

julia> algorithm3([0, 0, 0], 0.5)           # fill with 1 / length
3-element Vector{Float64}:
 0.3333333333333333
 0.3333333333333333
 0.3333333333333333

julia> algorithm3([0//1, 0//1], 0.0)        # fill with 1 / length, even if zero mass
2-element Vector{Rational{Int64}}:
 1//2
 1//2

julia> algorithm3([1, 2, 3], 0.9)           # in absence of 0's, just normalize
3-element Vector{Float64}:
 0.16666666666666666
 0.3333333333333333
 0.5
```
"""
algorithm3(p::Vector{T}, u::S) where {T<:Real, S<:Real} =
    (_check_u01(u); algorithm3!(similar(p, _typeofinv(T)), p, u))

#### Algorithm 3, in terms of ratio

"""
    algorithm3_ratio!(p::Vector{T}, r::Real) where {T<:Real}

Normalize `p` to probabilities, then spread the probability mass `u = r / (1 + r)`
across the 0 or more elements of `p` such that the ratio of (inititally) zero elements
to non-zero elements is equal to `r`.
Note that `T` must be a type which is able to hold the result of `inv(one(T))`.

See also: [`algorithm3_ratio`](@ref), [`algorithm3!`](@ref)
"""
algorithm3_ratio!(p, r) = algorithm3!(p, _u(r))

"""
    algorithm3_ratio!(p::Vector{T}, w::Vector{<:Real}, r::Real) where {T<:Real}

Normalize `w` to probabilities, storing the result in `p`, then spread the
probability mass `u = r / (1 + r)` across the 0 or more elements of `p` such that
the ratio of (inititally) zero elements to non-zero elements is equal to `r`.
Note that `T` must be a type which is able to hold the result of `inv(one(T))`.
"""
algorithm3_ratio!(p, w, r) = algorithm3!(p, w, _u(r))

"""
    algorithm3_ratio(w::Vector{<:Real}, r::Real)

Return a vector of probabilities by normalizing `w` to probabilities, then
spread the probability mass `u = r / (1 + r)` across the 0 or more elements of `w`
such that the ratio of (inititally) zero elements to non-zero elements is equal to `r`.

Mathematically, given:

𝐰 ∈ ℝᴺ, r ∈ ℝ₊, 0 ≤ r < Inf, J = {i : 𝐰ᵢ = 0}

```
pᵢ =
    Case 1: if J ≠ ∅
            (r / (1+r)) / |J|                     if i ∈ J
            (1 / (1+r)) * 𝐰ᵢ / ∑ᵢ₌₁ᴺ 𝐰ᵢ           otherwise
    Case 2: if J = {1,…,N}
            1/N
```

See also: [`algorithm3_ratio!`](@ref), [`algorithm3`](@ref)

# Examples
```jldoctest
julia> w = [1, 0, 3, 0, 5]; r = 2;

julia> algorithm3_ratio(w, r)
5-element Vector{Float64}:
 0.03703703703703704
 0.3333333333333333
 0.11111111111111113
 0.3333333333333333
 0.1851851851851852

julia> algorithm3(w, r / (1 + r))    # Note equivalence
5-element Vector{Float64}:
 0.03703703703703704
 0.3333333333333333
 0.11111111111111113
 0.3333333333333333
 0.1851851851851852
```
"""
algorithm3_ratio(p, r) = algorithm3(p, _u(r))

################

#### Algorithm 2.1. + Algorithm 3. (fused)

# A weight is assigned to i = 1,…,k components, and there are unknown components k+1,…,N.
# The unknown components are of the same category, and the probability mass of the category is
# known; alternatively, the ratio (between unknown/known) of probability masses may be specified.
# r = unknown/known = (∑ᵢ₌ₖ₊₁ᴺ pᵢ) / ∑ᵢ₌₁ᵏ pᵢ = (∑ᵢ₌ₖ₊₁ᴺ wᵢ) / ∑ᵢ₌₁ᵏ wᵢ ⟹
# r∑ᵢ₌₁ᵏ wᵢ = ∑ᵢ₌ₖ₊₁ᴺ wᵢ ⟹ r∑ᵢ₌₁ᵏ = w′, wᵢ = w′ / (N - k), i=k+1,…,N
# r = u / (1 - u) ⟹ u = r / (1 + r) ⟹
# pᵢ = u / (N - k), i=k+1,…,N
# pᵢ = (1 - u) wᵢ / ∑ᵢ₌₁ᵏ wᵢ, i = 1,…,k
"""
    algorithm2_1_algorithm3!(p::Vector{T}, I::Vector{Int}, w::Vector{<:Real}, u::Real) where {T<:Real}

Normalize `w[I]` to probabilities, storing the result in `p`, then spreading probability mass
`0 ≤ u ≤ 1` across the 0 or more elements of `w[I]` which are equal to zero.

Fill `p` with the probabilities which result from normalizing the weights selected by `I`
from `w`, wherein zero or more of the elements of `w` has an unknown (indicated by `0`) value.
The total probability mass of the unknown category is specified by `u`.
Caller must ensure that `u` is in the closed interval [0, 1].
If all selected values are zero, `p` is filled with `1 / length(p)`.
Note that `T` must be a type which is able to hold the result of `inv(one(T))`.

See also: [`algorithm2_1_algorithm3`](@ref)

# Examples
```jldoctest
julia> I = [1, 2, 5, 6]; w = [10, 0, 30, 40, 0, 20]; u = 0.5;

julia> algorithm2_1_algorithm3!(similar(I, Float64), I, w, u)
4-element Vector{Float64}:
 0.16666666666666666
 0.25
 0.25
 0.3333333333333333
```
"""
function algorithm2_1_algorithm3!(p::Vector{S}, I::Vector{Int}, w::Vector{T}, u::S) where {S<:Real, T<:Real}
    _check_u01(u)
    checkbounds(w, I)
    s = zero(T)
    z = 0
    @inbounds @simd ivdep for i ∈ eachindex(p, I)
        w̃ = w[I[i]]
        s += w̃
        p[i] = w̃
        z += w̃ == zero(T)
    end
    c = z == 0 ? one(S) / s : (one(S) - u) / s
    u′ = z == length(p) ? one(S) / z : u / z
    @inbounds @simd for i ∈ eachindex(p)
        pᵢ = p[i]
        p[i] = pᵢ == zero(S) ? u′ : pᵢ * c
        # p[i] = ifelse(pᵢ == zero(S), u′, pᵢ * c)
    end
    p
end
algorithm2_1_algorithm3!(p::Vector{S}, I::Vector{Int}, w::Vector{T}, u::U) where {S<:Real, T<:Real, U<:Real} = algorithm2_1_algorithm3!(p, I, w, convert(S, u))

"""
    algorithm2_1_algorithm3(I::Vector{Int}, w::Vector{<:Real}, u::Real)

Return a vector of probabilities, normalizing the components selected from `w` by the
index set `I`, then spreading the probability mass `0 ≤ u ≤ 1` across the 0 or more
selected elements which are equal to zero.
Equivalent to `algorithm3(algorithm2_1(I, w), u)` but more efficient.

See also: [`algorithm2_1_algorithm3!`](@ref)

# Examples
```jldoctest
julia> I = [1, 2, 5, 6]; w = [10, 0, 30, 40, 0, 20]; u = 0.5;

julia> algorithm2_1_algorithm3(I, w, u)
4-element Vector{Float64}:
 0.16666666666666666
 0.25
 0.25
 0.3333333333333333

julia> algorithm3(algorithm2_1(I, w), u)
4-element Vector{Float64}:
 0.16666666666666666
 0.25
 0.25
 0.3333333333333333
```
"""
algorithm2_1_algorithm3(I::Vector{Int}, w::Vector{T}, u::S) where {T<:Real, S<:Real} =
    algorithm2_1_algorithm3!(similar(I, _typeofinv(T)), I, w, u)

################
# Algorithm 4

# A weight is assigned to each i, and the w₁'s are normalized to probabilities.
# Then, a subset of the i's, denoted I′, is selected for re-weighting by a quantity
# which is undefined for I ∖ I′.
# w₁ ∈ ℝᴰ : the weight assigned to each i for the normalization of probabilities
# w₂ ∈ ℝᴺ : the quantity which is undefined for I ∖ I′; undefined shall be encoded
# by a value of zero.
# pᵢ = w₁ᵢ / ∑ₗ₌₁ᴺ w₁ₗ, i ∈ I ∖ I′
# mᵏⁿᵒʷⁿ = ∑ᵢ pᵢ, i ∈ I ∖ I′
# mᵘⁿᵈᵉᶠⁱⁿᵉᵈ = 1 - mᵏⁿᵒʷⁿ = (∑ᵢ w₁ᵢ, i ∈ I′) / ∑ₗ₌₁ᴺ w₁ₗ
# pᵢ = mᵘⁿᵈᵉᶠⁱⁿᵉᵈ * w₂ᵢ / ∑ₗ₌₁ᴺ w₂ₗ, i ∈ I′
# In other words,
# pᵢ = (w₂ᵢ * ∑ₗ w₁ₗ, i ∈ I ∖ I′) / (∑ₗ₌₁ᴺ w₂ₗ * ∑ₗ₌₁ᴺ w₁ₗ)

# A weight is assigned to each i, and the w₁'s are normalized to probabilities.
# Then, a subset of the i's, denoted I′, is selected for re-weighting by a quantity
# which is undefined for I ∖ I′.
# I = {1,…,N}
# J₁ = {i: 𝐰₁ᵢ = 0}    I₁′ = {i: 𝐰₁ᵢ ≠ 0} = I ∖ J₁
# J₂ = {i: 𝐰₂ᵢ = 0}    I₂′ = {i: 𝐰₂ᵢ ≠ 0} = I ∖ J₂
# 𝐰₁ ∈ ℝᴺ : the initial weights
# 𝐰₂ ∈ ℝᴺ : the quantity which is undefined for J₂ = I ∖ I₂′; undefined shall be encoded
# by a value of zero in 𝐰₂.
# pᵢ = 𝐰₁ᵢ / ∑ₗ₌₁ᴺ 𝐰₁ₗ, i ∈ I ∖ I₂′
# mᵏⁿᵒʷⁿ = ∑ᵢ pᵢ, i ∈ I ∖ I₂′
# mᵘⁿᵈᵉᶠⁱⁿᵉᵈ = 1 - mᵏⁿᵒʷⁿ = (∑ᵢ 𝐰₁ᵢ, i ∈ I₂′) / ∑ₗ₌₁ᴺ 𝐰₁ₗ
# pᵢ = mᵘⁿᵈᵉᶠⁱⁿᵉᵈ * 𝐰₂ᵢ / ∑ₗ₌₁ᴺ 𝐰₂ₗ, i ∈ I₂′
# In other words,
# pᵢ = (𝐰₂ᵢ * ∑ₗ 𝐰₁ₗ, l ∈ I₂′) / (∑ₗ₌₁ᴺ 𝐰₂ₗ * ∑ₗ₌₁ᴺ 𝐰₁ₗ)    i ∈ I₂′
## As cases, for clarity
# pᵢ = 𝐰₁ᵢ / ∑ₗ₌₁ᴺ 𝐰₁ₗ                                      i ∈ I ∖ I₂′
# pᵢ = (𝐰₂ᵢ * ∑ₗ 𝐰₁ₗ, l ∈ I₂′) / (∑ₗ₌₁ᴺ 𝐰₂ₗ * ∑ₗ₌₁ᴺ 𝐰₁ₗ)    i ∈ I₂′
# general, but must be protected against 𝐰₁ = ̲0 and/or 𝐰₂ = ̲0, which cause /0 error.
# Essentially, if
#     s₁ = ∑ₗ₌₁ᴺ 𝐰₁ₗ
#     s₂ = ∑ₗ₌₁ᴺ 𝐰₂ₗ
# and if s₁ = 0, then s₁ must be set equal to 1 to keep the terms defined.
# The same argument applies to s₂.
# An alternative line of reasoning suggests that it is preferable to be
# mathematically consistent and let /0 cause the expected behavior (NaNs).
# Mathematical consistency is much easier to reason about, as the definition
# of the algorithm clearly implies that if 𝐰₁ = ̲0, then everything that follows
# involves division by 0.

# _c1c2(::Type{T}, s₁′, s₁, s₂) where {T} = convert(T, inv(s₁)), convert(T, s₁′ / (s₁ * s₂))

"""
    algorithm4!(𝐰₁::Vector{T}, 𝐰₂::Vector{<:Real}) where {T<:Real}

Fill `𝐰₁` with the probabilities which result from `algorithm4(𝐰₁, 𝐰₂)`; refer to the
respective documentation for a description of `algorithm4`.
Note that `T` must be a type which is able to hold the result of `inv(one(T))`.

See also: [`algorithm4`](@ref)
"""
function algorithm4!(w₁::Vector{T}, w₂::Vector{U}) where {T<:Real, U<:Real}
    s₁′ = zero(T)
    s₁ = zero(T)
    s₂ = zero(U)
    @inbounds @simd for i ∈ eachindex(w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        s₁′ += w₂ᵢ == zero(U) ? zero(T) : w₁ᵢ
        s₁ += w₁ᵢ
        s₂ += w₂ᵢ
    end
    c₁ = inv(s₁)
    c₂ = s₁′ / (s₁ * s₂)
    # Unlike below, the potential instability is unavoidable here.
    # c₁, c₂ = _c1c2(T, s₁′, s₁, s₂)
    @inbounds @simd for i ∈ eachindex(w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        w₁[i] = w₂ᵢ == zero(U) ? c₁ * w₁ᵢ : c₂ * w₂ᵢ
    end
    w₁
end

"""
    algorithm4!(p::Vector{T}, 𝐰₁::Vector{<:Real}, 𝐰₂::Vector{<:Real}) where {T<:Real}

Fill `p` with the probabilities which result from `algorithm4(𝐰₁, 𝐰₂)`; refer to the
respective documentation for a description of `algorithm4`.
Note that `T` must be a type which is able to hold the result of `inv(one(T))`.

See also: [`algorithm4`](@ref)
"""
function algorithm4!(p::Vector{S}, w₁::Vector{T}, w₂::Vector{U}) where {S<:Real, T<:Real, U<:Real}
    s₁′ = zero(T)
    s₁ = zero(T)
    s₂ = zero(U)
    @inbounds @simd for i ∈ eachindex(w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        s₁′ += w₂ᵢ == zero(U) ? zero(T) : w₁ᵢ
        s₁ += w₁ᵢ
        s₂ += w₂ᵢ
    end
    # c₁ = inv(s₁)
    # c₂ = s₁′ / (s₁ * s₂)
    # Equivalent, but improves type stability at expensive of inv(::Rational) not being used.
    # Note, however, inv(::Rational) occurs at most once, whereas the instability in the loop
    # incurs overhead length(p) times.
    c₁ = one(S) / s₁
    c₂ = s₁′ * c₁ / s₂
    @inbounds @simd for i ∈ eachindex(p, w₁, w₂)
        w₁ᵢ = w₁[i]
        w₂ᵢ = w₂[i]
        p[i] = w₂ᵢ == zero(U) ? c₁ * w₁ᵢ : c₂ * w₂ᵢ
    end
    p
end

"""
    algorithm4(𝐰₁::Vector{<:Real}, 𝐰₂::Vector{<:Real})

Return a vector of probabilities by constructed according to the following algorithm:

Define:

I = {1,…,N}

J₁ = {i: 𝐰₁ᵢ = 0},    I₁′ = {i: 𝐰₁ᵢ ≠ 0} = I ∖ J₁

J₂ = {i: 𝐰₂ᵢ = 0},    I₂′ = {i: 𝐰₂ᵢ ≠ 0} = I ∖ J₂

𝐰₁ ∈ ℝᴺ : initial weights, 0 ≤ 𝐰₁ᵢ < Inf

𝐰₂ ∈ ℝᴺ : augment weights, 0 ≤ 𝐰₂ᵢ < Inf; a value of zero indicates no re-weight

Then:

pᵢ = 𝐰₁ᵢ / ∑ₗ₌₁ᴺ 𝐰₁ₗ,                                       i ∈ I ∖ I₂′

pᵢ = (𝐰₂ᵢ * ∑ₗ 𝐰₁ₗ, l ∈ I₂′) / (∑ₗ₌₁ᴺ 𝐰₂ₗ * ∑ₗ₌₁ᴺ 𝐰₁ₗ),     i ∈ I₂′

This algorithm can produce a wide variety of probability vectors as the result
of the various combinations of intersections which can be formed from J₁, J₂, I₁′, and I₂′.
However, complexity of outputs aside, the motivating concept is quite simple:
take a vector of weights, `𝐰₁` and re-weight some subset (I₂′) of those weights using
a second set of weights, `𝐰₂`, while preserving the proportion of probability mass
derived from `𝐰₁`. That is, given `p = algorithm4(𝐰₁, 𝐰₂)`, the following relationship
is preserved: `sum(p[J₂]) ≈ sum(𝐰₁[J₂]) / sum(𝐰₁[I₁′])`.

See also: [`algorithm4!`](@ref)

# Examples
```jldoctest
julia> w₁ = [1, 1, 1, 1, 0];

julia> algorithm4(w₁, [2, 1, 3, 4, 0])    # J₁ ∩ I₂′ = ∅
5-element Vector{Float64}:
 0.2
 0.1
 0.30000000000000004
 0.4
 0.0

julia> algorithm4(w₁, [2, 1, 3, 0, 5])    # J₂ = [4] not re-weighted; I₂′ re-weighted
5-element Vector{Float64}:
 0.13636363636363635
 0.06818181818181818
 0.20454545454545453
 0.25
 0.3409090909090909

julia> w₁ = [1, 1, 1, 0, 0];

julia> algorithm4(w₁, [2, 1, 3, 4, 0])    # J₂ = [5] not re-weighted; I₂′ re-weighted
5-element Vector{Float64}:
 0.2
 0.1
 0.30000000000000004
 0.4
 0.0

julia> w₁ = [1, 1, 0, 1, 0];

julia> algorithm4(w₁, [0, 1, 0, 4, 0])    # J₂ = [1,3,5] not re-weighted; I₂′ re-weighted
5-element Vector{Float64}:
 0.3333333333333333
 0.13333333333333333
 0.0
 0.5333333333333333
 0.0

julia> algorithm4(w₁, [0, 0, 3, 4, 0])    # J₂ = [1,2,5] not re-weighted; I₂′ re-weighted
5-element Vector{Float64}:
 0.3333333333333333
 0.3333333333333333
 0.14285714285714285
 0.19047619047619047
 0.0

julia> algorithm4(w₁, [2, 0, 3, 0, 0])    # J₂ = [2,4,5] not re-weighted; I₂′ re-weighted
5-element Vector{Float64}:
 0.13333333333333333
 0.3333333333333333
 0.2
 0.3333333333333333
 0.0
```
"""
algorithm4(w₁::Vector{T}, w₂::Vector{U}) where {T<:Real, U<:Real} = algorithm4!(similar(w₁, promote_type(_typeofinv(T), _typeofinv(U))), w₁, w₂)
