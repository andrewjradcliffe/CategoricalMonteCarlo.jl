module CategoricalMonteCarlo

using Random, Test, BenchmarkTools
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
    algorithm3, algorithm3!, algorithm4!, algorithm4,
    normalize1, normalize1!

include("utils.jl")
include("normalizations.jl")
include("sampler.jl")
include("tsampler_batch.jl")
include("vsampler.jl")
include("vtsampler_batch.jl")

end
