# CategoricalMonteCarlo

## Installation
```julia
using Pkg
Pkg.add("CategoricalMonteCarlo")
```

## Description

Given `i=1,…,n` independent categorical distributions, each with a
unique probability mass vector, `pᵢ`, what is the distribution of the
sum of the joint distribution formed from the product of the
marginals? Assume that some categories are shared across the
marginals, such that by re-indexing and constructing a sparse
representation based on the original probability mass vectors, we may
unify the categories themselves. However, the sum of the joint
distribution would not be multinomial unless we have the trivial case
of each distribution being identical. While there is no closed-form
expression for the sum of the joint distribution, Monte Carlo
simulation provides a general mechanism for computation.

This package provides the facilities for such Monte Carlo simulation,
based on collections of probability mass vectors, each of which
corresponds to a (possibly, but not necessarily) independent
categorical distribution. Several advanced strategies are utilized to
maximize the performance of such computations, including
fastest-in-Julia categorical sampling (`MarsagliaDiscreteSamplers`) --
in comparison to publicly visible packages -- in addition to
partitioning strategies which favor memory locality and cache
performance despite the random-access nature writes inherent to Monte
Carlo simulation. These same partitioning strategies are utilized to
enable thread-based parallelism across the iteration space of
arbitrary-dimensional input arrays. Furthermore, reduction-in-place is
supported via the interface familiar to Julia users -- the
`dims::Vararg{<:Integer, N} where N` keyword; this enables additional
increases in efficiency, as while the user may wish to simulate a
distribution bearing the indices of the input array, it may be known
that some of these dimensions will always be summed over.

## Usage

It may help to demonstrate with an example. Consider an
equally-weighted coin with sides labeled 1 and 2; an equally-weighted
four-sided die with sides labeled 1, 2, 3 and 4; an equally-weighted
six-sided die with sides labeled 1, 2, 3, 4, 5 and 6. If one were
consider a scenario in which one flips the coin, rolls the four-sided
die, and rolls the six-sided die, what is the distribution of counts
on labels 1,...,6?

```julia
julia> using CategoricalMonteCarlo

julia> coin = [1/2, 1/2];

julia> die4 = [1/4, 1/4, 1/4, 1/4];

julia> die6 = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

julia> sample([coin, die4, die6], 10)
10×6 Matrix{Int64}:
 0  1  0  1  0  1
 1  1  1  0  0  0
 1  1  0  0  1  0
 0  1  1  1  0  0
 1  0  0  1  0  1
 0  2  0  0  0  1
 1  1  1  0  0  0
 1  0  0  1  0  1
 3  0  0  0  0  0
 1  1  0  1  0  0

julia> using MonteCarloSummary

julia> ["mean" "mcse" "std" "2.5th" "50th" "97.5"; 
        mcsummary(sample([coin, die4, die6], 10^6), (0.025, 0.5, 0.975))]
7×6 Matrix{Any}:
  "mean"    "mcse"       "std"     "2.5th"   "50th"   "97.5"
 0.918372  0.000759618  0.759618  0.0       1.0      2.0
 0.915886  0.000758815  0.758815  0.0       1.0      2.0
 0.41628   0.000571198  0.571198  0.0       0.0      2.0
 0.416611  0.000570929  0.570929  0.0       0.0      2.0
 0.166285  0.000372336  0.372336  0.0       0.0      1.0
 0.166566  0.000372588  0.372588  0.0       0.0      1.0
```


## Future Work
- Description of normalizers (weight vector->probability vector)
