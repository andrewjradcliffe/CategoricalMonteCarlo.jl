#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

function sample1(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., n_sim)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample1!(B, A, dims)
end

function sample1!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {T<:AbstractFloat, N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    for j ∈ axes(B, N′)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for (Iₛ, ω) ∈ a
                c = categorical(ω)
                B[Iₛ[c], IR, j] += one(S)
            end
        end
    end
    B
end


function sample1(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., n_sim)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample1!(B, A, dims)
end

function sample1!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    for j ∈ axes(B, N′)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for Iₛ ∈ a
                c = rand(Iₛ)
                B[c, IR, j] += one(S)
            end
        end
    end
    B
end

function tsample1(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., n_sim)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    tsample!(B, A, dims)
end

function tsample1!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    Threads.@threads for j ∈ axes(B, N′)
        for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for Iₛ ∈ a
                c = rand(Iₛ)
                B[c, IR, j] += one(S)
            end
        end
    end
    B
end


# As the input array becomes large, SIMD PRNG sampling tends to be better
# due to the fact that each element of A is accessed only once.
# -- There is always the option of sampling across the j-indices of B
# and placing dimensions of A on the 3rd...end positions.
# If annotated with @inbounds and @simd, this is as fast (or faster) than
# the simple `sample_simd` approach.

function sample2(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample2!(B, A, dims)
end

function sample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {T<:AbstractFloat, N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    C = Vector{Int}(undef, size(B, 2))
    U = Vector{Float64}(undef, size(B, 2))
    Σp = Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, ω) ∈ a
            resize!(Σp, length(ω))
            cumsum!(Σp, ω)
            categorical!(C, U, Σp)
            @simd for j ∈ axes(B, 2) # ArrayInterface.indices((B, C), (2, 1))
                c = C[j]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

function sample2(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample2!(B, A, dims)
end

function sample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    C = Vector{Int}(undef, size(B, 2))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            rand!(C, Iₛ)
            @simd for j ∈ axes(B, 2)
                c = C[j]
                B[c, j, IR] += one(S)
            end
        end
    end
    B
end

function tsample2(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, n_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    tsample2!(B, A, dims)
end

function tsample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    tsample2!(B, A, keep, default, firstindex(B, 2):size(B, 2))
end

function tsample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L) # similar(𝒥, Int)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for Iₛ ∈ a
                rand!(C, Iₛ)
                @simd for l ∈ eachindex(C, 𝒥)
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
            Threads.@spawn tsample2!(B, A, keep, default, start:h)
            tsample2!(B, A, keep, default, (h + 1):stop)
        end
        return B
    end
end

function sample3(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., n_sim)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample3!(B, A, dims)
end

function sample3!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    C = Vector{Int}(undef, size(B, N′))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            rand!(C, Iₛ)
            @simd for j ∈ axes(B, N′)
                c = C[j]
                B[c, IR, j] += one(S)
            end
        end
    end
    B
end

function tsample3(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., n_sim)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    tsample3!(B, A, dims)
end

function tsample3!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    tsample3!(B, A, keep, default, firstindex(B, N′):size(B, N′))
end

function tsample3!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        C = Vector{Int}(undef, L) # similar(𝒥, Int)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for Iₛ ∈ a
                rand!(C, Iₛ)
                @simd for l ∈ eachindex(C, 𝒥)
                    c = C[l]
                    j = 𝒥[l]
                    B[c, IR, j] += one(S)
                end
            end
        end
        return B
    else
        h = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample3!(B, A, keep, default, start:h)
            tsample3!(B, A, keep, default, (h + 1):stop)
        end
        return B
    end
end

################
# Sampler which has simulation index on first dimension, categories on second dimension
# Follows sample2's convention otherwise
# Alas, this does not make much difference.
# While only a few categories would potentially occur on each rand! + increment
# (the innermost body + loop), these categories are random, hence, the instruction pipeline
# is most likely unable to do useful prediction. Finding the appropriate columns
# myself would not be very useful either, as this would require a view (which costs)
# in addition to other costs.
function sample4(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_sim, n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample4!(B, A, dims)
end

function sample4!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {T<:AbstractFloat, N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    C = Vector{Int}(undef, size(B, 1))
    U = Vector{Float64}(undef, size(B, 1))
    Σp = Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, ω) ∈ a
            resize!(Σp, length(ω))
            cumsum!(Σp, ω)
            categorical!(C, U, Σp)
            @simd for j ∈ axes(B, 1) # ArrayInterface.indices((B, C), (2, 1))
                c = C[j]
                B[j, Iₛ[c], IR] += one(S)
            end
        end
    end
    B
end

function sample4(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, n_sim::Int, n_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(n_sim, n_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
    B = similar(A, S, Dᴮ)
    fill!(B, zero(S))
    sample4!(B, A, dims)
end

function sample4!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}, dims::NTuple{P, Int}) where {S<:Real, N′} where {P} where {N}
    keep = ntuple(d -> d ∉ dims, Val(N))
    default = ntuple(d -> firstindex(A, d), Val(N))
    C = Vector{Int}(undef, size(B, 1))
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            rand!(C, Iₛ)
            @simd for j ∈ axes(B, 1)
                c = C[j]
                B[j, c, IR] += one(S)
            end
        end
    end
    B
end

################
# Non-allocating versions

# Oddly, the performance of the non-allocating variants differs on the
# array of vector / array of array of vector cases (compared to the single sparse vector case).
# As the arrays become large, O(10^4), for the Vector{Int} cases,
# the temporary array is faster by 20-25%. For the Tuple{Vector{Int}, Vector{<:AbstractFloat}}
# case, the temporary array is 10% slower.
# This needs more extensive benchmarking to determine which is optimal -- the answer
# very likely depends on the scale of the problem (n_sim and n_cat) and very likely
# the distributions of probability mass (more uniform being worse?).

function sample0!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    Σω = Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, ω) ∈ a
            resize!(Σω, length(ω))
            cumsum!(Σω, ω)
            for j ∈ axes(B, 2)
                c = rand_invcdf(Σω)
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

function sample0!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    Σω = Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, ω = A[IA]
        resize!(Σω, length(ω))
        cumsum!(Σω, ω)
        for j ∈ axes(B, 2)
            c = rand_invcdf(Σω)
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

function sample0!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for Iₛ ∈ a
            for j ∈ axes(B, 2)
                c = rand(Iₛ)
                B[c, j, IR] += one(S)
            end
        end
    end
    B
end


################
# Limiting chunksize

# The smaller chunksize approach actually performs ≈5-8% worse for both
# the equiprobable and nonequiprobable cases.
function sample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    Σω = Vector{T}()
    q, r = divrem(size(B, 2), 1024)
    if q == 0
        C = Vector{Int}(undef, r)
        U = Vector{Float64}(undef, r)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for (Iₛ, ω) ∈ a
                resize!(Σω, length(ω))
                cumsum!(Σω, ω)
                categorical!(C, U, Σω)
                for j ∈ axes(B, 2)
                    c = C[j]
                    B[Iₛ[c], j, IR] += one(S)
                end
            end
        end
    else
        C = Vector{Int}(undef, 1024)
        U = Vector{Float64}(undef, 1024)
        ax = axes(B, 2)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for (Iₛ, ω) ∈ a
                resize!(C, 1024)
                resize!(U, 1024)
                resize!(Σω, length(ω))
                cumsum!(Σω, ω)
                J = iterate(ax)
                for _ = 1:q
                    categorical!(C, U, Σω)
                    for c ∈ C
                        j, js = J
                        B[Iₛ[c], j, IR] += one(S)
                        J = iterate(ax, j)
                    end
                end
                if r != 0
                    resize!(C, r)
                    resize!(U, r)
                    categorical!(C, U, Σω)
                    for c ∈ C
                        j, js = J
                        B[Iₛ[c], j, IR] += one(S)
                        J = iterate(ax, j)
                    end
                end
            end
        end
    end
    B
end

function sample2!(B::AbstractArray{S, N′}, A::AbstractArray{Vector{Vector{Int}}, N}) where {S<:Real, N′} where {N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    q, r = divrem(size(B, 2), 1024)
    if q == 0
        C = Vector{Int}(undef, r)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for Iₛ ∈ a
                rand!(C, Iₛ)
                for j ∈ axes(B, 2)
                    c = C[j]
                    B[c, j, IR] += one(S)
                end
            end
        end
    else
        C = Vector{Int}(undef, 1024)
        ax = axes(B, 2)
        @inbounds for IA ∈ CartesianIndices(A)
            IR = Broadcast.newindex(IA, keep, default)
            a = A[IA]
            for Iₛ ∈ a
                resize!(C, 1024)
                J = iterate(ax)
                for _ = 1:q
                    rand!(C, Iₛ)
                    for c ∈ C
                        j, js = J
                        B[c, j, IR] += one(S)
                        J = iterate(ax, j)
                    end
                end
                if r != 0
                    resize!(C, r)
                    rand!(C, Iₛ)
                    for c ∈ C
                        j, js = J
                        B[c, j, IR] += one(S)
                        J = iterate(ax, j)
                    end
                end
            end
        end
    end
    B
end

@inline function _unsafe_sample!(B::AbstractArray{S}, Iₛ, Σω, U, ax, J, k, s₀) where {S<:Real}
    @inbounds for i ∈ eachindex(U)
        j, js = J
        u = U[i]
        c = 1
        s = s₀
        while s < u && c < k
            c += 1
            s = Σω[c]
        end
        B[Iₛ[c], j] += one(S)
        J = iterate(ax, j)
    end
    J
end

# limiting the chunksize of U
# Other than saving on the memory allocation, this is equivalent speed to the simpler method.
function sample2!(B::AbstractArray{S, N′}, A::Tuple{Vector{Int}, Vector{T}}) where {S<:Real, N′} where {T<:AbstractFloat}
    Iₛ, ω = A
    Σω = cumsum(ω)
    k = length(ω)
    s₀ = Σω[1]
    q, r = divrem(size(B, 2), 1024)
    if q == 0
        U = rand(r)
        @inbounds for j ∈ axes(B, 2)
            u = U[j]
            c = 1
            s = s₀
            while s < u && c < k
                c += 1
                s = Σω[c]
            end
            B[Iₛ[c], j] += one(S)
        end
    else
        U = Vector{Float64}(undef, 1024)
        ax = axes(B, 2)
        J = iterate(ax)
        for _ = 1:q
            rand!(U)
            J = _unsafe_sample!(B, Iₛ, Σω, U, ax, J, k, s₀)
        end
        if r != 0
            resize!(U, r)
            rand!(U)
            _unsafe_sample!(B, Iₛ, Σω, U, ax, J, k, s₀)
        end
    end
    B
end

################
# Marsaglia square histogram method

# The expected case: vectors of sparse vectors (as their bare components)
function sample_mars!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}) where {S<:Real, N′} where {R<:AbstractArray{Tuple{Vector{Int}, Vector{T}}, M}, N} where {T<:AbstractFloat, M}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    K, V = Vector{Int}(), Vector{T}()
    ix, q = Vector{Int}(), Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        a = A[IA]
        for (Iₛ, ω) ∈ a
            n = length(ω)
            resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
            marsaglia!(K, V, q, ix, ω)
            marsaglia_generate!(C, K, V)
            for j ∈ axes(B, 2)
                c = C[j]
                B[Iₛ[c], j, IR] += one(S)
            end
        end
    end
    B
end

# A simplification: an array of sparse vectors
function sample_mars!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 2))
    K, V = Vector{Int}(), Vector{T}()
    ix, q = Vector{Int}(), Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, ω = A[IA]
        n = length(ω)
        resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
        marsaglia!(K, V, q, ix, ω)
        marsaglia_generate!(C, K, V)
        for j ∈ axes(B, 2)
            c = C[j]
            B[Iₛ[c], j, IR] += one(S)
        end
    end
    B
end

function sample_mars_dim1_4!(B::AbstractArray{S, N′}, A::AbstractArray{Tuple{Vector{Int}, Vector{T}}, N}) where {S<:Real, N′} where {T<:AbstractFloat, N}
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    C = Vector{Int}(undef, size(B, 1))
    U = Vector{Float64}(undef, size(B, 1))
    K, V = Vector{Int}(), Vector{T}()
    ix, q = Vector{Int}(), Vector{T}()
    @inbounds for IA ∈ CartesianIndices(A)
        IR = Broadcast.newindex(IA, keep, default)
        Iₛ, ω = A[IA]
        n = length(ω)
        resize!(K, n); resize!(V, n); resize!(ix, n); resize!(q, n)
        marsaglia!(K, V, q, ix, ω)
        marsaglia_generate5!(C, U, K, V)
        for j ∈ axes(B, 1)
            c = C[j]
            B[j, Iₛ[c], IR] += one(S)
        end
        # @inbounds for j ∈ axes(B, 1)
        #     u = rand()
        #     j′ = floor(Int, muladd(u, n, 1))
        #     c = u < V[j′] ? j′ : K[j′]
        #     B[j, Iₛ[c], IR] += one(S)
        # end
    end
    B
end

# The simplest case: a sparse vector
sample_mars(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::Int) where {S<:Real} where {T<:AbstractFloat} = sample_mars(S, A, n_sim, n_cat, :)
sample_mars(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, dims::NTuple{N, Int}) where {S<:Real} where {T<:AbstractFloat} where {N} = sample_mars(S, A, n_sim, n_cat, :)

function sample_mars(::Type{S}, A::Tuple{Vector{Int}, Vector{T}}, n_sim::Int, n_cat::Int, ::Colon) where {S<:Real} where {T<:AbstractFloat} where {N}
    B = zeros(S, n_cat, n_sim)
    sample_mars!(B, A)
end

function sample_mars!(B::AbstractMatrix{S}, A::Tuple{Vector{Int}, Vector{T}}) where {S<:Real} where {T<:AbstractFloat}
    _check_reducedims(B, A)
    Iₛ, ω = A
    # k = length(ω)
    # Σω = cumsum(ω)
    # s₀ = Σω[1]
    K, V = marsaglia(ω)
    n = length(K)
    @inbounds for j ∈ axes(B, 2)
        u = rand()
        j′ = floor(Int, muladd(u, n, 1))
        c = u < V[j′] ? j′ : K[j′]
        B[Iₛ[c], j] += one(S)
    end
    # C = Vector{Int}(undef, size(B, 2))
    # # marsaglia_generate!(C, K, V)
    # marsaglia_generate4!(C, K, V)
    # @inbounds for j ∈ eachindex(axes(B, 2), C)
    #     B[Iₛ[C[j]], j] += one(S)
    # end
    B
end

#### Threading experiments
function _vtsample_chunk!(B::AbstractArray{S, N′}, A::AbstractArray{R, N}, keep, default, 𝒥::UnitRange{Int}) where {S<:Real, N′} where {R<:AbstractArray{Vector{Int}, M}, N} where {M}
    L = length(𝒥)
    C = Vector{Int}(undef, L)
    U = Vector{Float64}(undef, L)
    K, V = Vector{Int}(), Vector{Float64}()
    ix, q = Vector{Int}(), Vector{Float64}()
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

function vtsample_poly!(B, A, sz::Int)
    _check_reducedims(B, A)
    keep, default = Broadcast.shapeindexer(axes(B)[3:end])
    rs = splitranges(firstindex(B, 2):size(B, 2), sz)
    @batch for r in rs
        _vtsample_chunk!(B, A, keep, default, r)
    end
    B
end



################
# Reference sampler which is as simple as possible.
function sample_simd!(B::Matrix{T}, A::Vector{Vector{Int}}) where {T<:Real}
    c = Vector{Int}(undef, size(B, 2))
    @inbounds for m ∈ eachindex(A)
        Iₛ = A[m]
        if length(Iₛ) == 1
            @inbounds i = Iₛ[1]
            @inbounds @simd ivdep for j ∈ axes(B, 2)
                B[i, j] += one(T)
            end
        else
            rand!(c, Iₛ)
            @inbounds @simd for j ∈ axes(B, 2)
                i = c[j]
                B[i, j] += one(T)
            end
        end
    end
    return B
end
sample_simd(::Type{T}, A::Vector{Vector{Int}}, I::Int, J::Int) where {T<:Real} =
    sample_simd!(zeros(T, I, J), A)

function sample_simd!(B::Matrix{T}, A::Array{Vector{Vector{Int}}, N}) where {T<:Real} where {N}
    c = Vector{Int}(undef, size(B, 2))
    @inbounds for n ∈ eachindex(A)
        a = A[n]
        for m ∈ eachindex(a)
            Iₛ = a[m]
            if length(Iₛ) == 1
                @inbounds i = Iₛ[1]
                @inbounds @simd ivdep for j ∈ axes(B, 2)
                    B[i, j] += one(T)
                end
            else
                rand!(c, Iₛ)
                @inbounds @simd for j ∈ axes(B, 2)
                    i = c[j]
                    B[i, j] += one(T)
                end
            end
        end
    end
    return B
end
sample_simd(::Type{T}, A::Array{Vector{Vector{Int}}, N}, I::Int, J::Int) where {T<:Real, N} =
    sample_simd!(zeros(T, I, J), A)


function tsample_simd!(B::Matrix{T}, A::Vector{Vector{Int}}, 𝒥::UnitRange{Int}) where {T<:Real}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        c = Vector{Int}(undef, L)
        @inbounds for m ∈ eachindex(A)
            Iₛ = A[m]
            if length(Iₛ) == 1
                @inbounds i = Iₛ[1]
                @inbounds @simd ivdep for j ∈ 𝒥
                    B[i, j] += one(T)
                end
            else
                rand!(c, Iₛ)
                @inbounds @simd for l ∈ eachindex(𝒥)
                    i = c[l]
                    B[i, 𝒥[l]] += one(T)
                end
            end
        end
        return B
    else
        H = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample_simd!(B, A, start:H)
            tsample_simd!(B, A, (H + 1):stop)
        end
        return B
    end
    return B
end
tsample_simd!(B::Matrix{T}, A::Vector{Vector{Int}}) where {T<:Real} = tsample_simd!(B, A, 1:size(B, 2))

tsample_simd(::Type{T}, A::Vector{Vector{Int}}, I::Int, J::Int) where {T<:Real} =
    tsample_simd!(zeros(T, I, J), A, 1:J)

function tsample_simd!(B::Matrix{T}, A::Array{Vector{Vector{Int}}, N}, 𝒥::UnitRange{Int}) where {T<:Real} where {N}
    (; start, stop) = 𝒥
    L = stop - start + 1
    if L ≤ 1024
        c = Vector{Int}(undef, L)
        @inbounds for n ∈ eachindex(A)
            a = A[n]
            for m ∈ eachindex(a)
                Iₛ = a[m]
                if length(Iₛ) == 1
                    @inbounds i = Iₛ[1]
                    @inbounds @simd ivdep for j ∈ 𝒥
                        B[i, j] += one(T)
                    end
                else
                    rand!(c, Iₛ)
                    @inbounds @simd for l ∈ eachindex(𝒥)
                        i = c[l]
                        B[i, 𝒥[l]] += one(T)
                    end
                end
            end
        end
        return B
    else
        H = (start + stop) >> 1
        @sync begin
            Threads.@spawn tsample_simd!(B, A, start:H)
            tsample_simd!(B, A, (H + 1):stop)
        end
        return B
    end
end
tsample_simd!(B::Matrix{T}, A::Array{Vector{Vector{Int}}, N}) where {T<:Real} where {N} =
    tsample_simd!(B, A, 1:size(B, 2))
tsample_simd(::Type{T}, A::Array{Vector{Vector{Int}}, N}, I::Int, J::Int) where {T<:Real} where {N} = tsample_simd!(zeros(T, I, J), A, 1:J)
