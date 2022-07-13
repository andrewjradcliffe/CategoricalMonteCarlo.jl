using CategoricalMonteCarlo
using Random
using SparseArrays
using Test

import CategoricalMonteCarlo: _check_reducedims, splitranges,
    _typeofinv, _typeofprod, _u, _check_u01

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

