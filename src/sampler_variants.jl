#
# Date created: 2022-06-12
# Author: aradclif
#
#
############################################################################################

function sample1(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_sim)
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


function sample1(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_sim)
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

function tsample1(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_sim)
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

function sample2(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, num_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
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

function sample2(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, num_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
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

function tsample2(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, num_sim, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
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

function sample3(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_sim)
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

function tsample3(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))..., num_sim)
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
function sample4(::Type{S}, A::AbstractArray{Vector{Tuple{Vector{Int}, Vector{T}}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {T<:AbstractFloat, N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_sim, num_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
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

function sample4(::Type{S}, A::AbstractArray{Vector{Vector{Int}}, N}, num_sim::Int, num_cat::Int, dims::NTuple{P, Int}) where {S<:Real} where {P} where {N}
    Dᴬ = size(A)
    Dᴮ = tuple(num_sim, num_cat, ntuple(d -> d ∈ dims ? 1 : Dᴬ[d], Val(N))...)
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
