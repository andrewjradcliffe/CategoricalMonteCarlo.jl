module CategoricalMonteCarlo

# using Test, BenchmarkTools
using Random
using SparseArrays
using LoopVectorization
using Polyester
using MarsagliaDiscreteSamplers

import MarsagliaDiscreteSamplers: _sqhist_init

export num_cat
export sample, sample!, tsample, tsample!
export vsample, vsample!, vtsample, vtsample!

# Ugly normalization names
export algorithm2_1, algorithm2_1!, algorithm2_2, algorithm2_2!,
    algorithm3, algorithm3!, algorithm3_ratio, algorithm3_ratio!,
    algorithm2_1_algorithm3, algorithm2_1_algorithm3!,
    algorithm2_1_algorithm3_ratio, algorithm2_1_algorithm3_ratio!,
    algorithm4!, algorithm4,
    normalize1, normalize1!

export pvg, pvg!, tpvg, tpvg!

include("utils.jl")
include("normalizations.jl")
include("sampler.jl")
include("tsampler_batch.jl")
include("vsampler.jl")
include("vtsampler_batch.jl")
include("probabilityvectorgeneration.jl")

end
