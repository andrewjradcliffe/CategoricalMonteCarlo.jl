using CategoricalMonteCarlo
using Test, Random

# @testset "CategoricalMonteCarlo.jl" begin
# end
const tests = [
    "utils.jl",
    "sampler.jl",
    "tsampler.jl",
]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end

