using CategoricalMonteCarlo
using Random
using Test

import CategoricalMonteCarlo: _check_reducedims, splitranges

const tests = [
    "utils.jl",
    "normalizations.jl",
    "sampler.jl",
    "tsampler_batch.jl",
    "vsampler.jl",
    "vtsampler_batch.jl",
]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end

