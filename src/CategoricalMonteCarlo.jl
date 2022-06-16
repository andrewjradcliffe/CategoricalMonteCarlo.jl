module CategoricalMonteCarlo

using Random, Test, BenchmarkTools

export sample, sample!, tsample, tsample!, num_cat, categorical, categorical!

include("utils.jl")
include("bernoulliprocess.jl")
include("sampler.jl")
include("tsampler.jl")

end
