module CategoricalMonteCarlo

using Random, Test, BenchmarkTools
using SparseArrays
using LoopVectorization
using Polyester
# using VectorizedReduction
using MarsagliaDiscreteSamplers

import MarsagliaDiscreteSamplers: _sqhist_init

export sample, sample!, tsample, tsample!, num_cat, categorical, categorical!

include("utils.jl")
include("normalizations.jl")
include("sampler.jl")
include("tsampler_batch.jl")
include("vsampler.jl")
include("vtsampler_batch.jl")

end
