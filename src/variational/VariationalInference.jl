module Variational

using ..Core, ..Core.RandomVariables, ..Utilities
using Distributions, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS
using ..Turing: Model, SampleFromPrior, SampleFromUniform
using ..Turing: Turing
using Random: AbstractRNG

using ForwardDiff
using Flux.Tracker
using Flux.Optimise


import ..Core: getchunksize, getADtype

export
    vi,
    ADVI,
    BBVI,
    ELBO


abstract type VariationalInference{AD} end

getchunksize(::T) where {T <: VariationalInference} = getchunksize(T)
getchunksize(::Type{<:VariationalInference{AD}}) where AD = getchunksize(AD)
getADtype(alg::VariationalInference) = getADtype(typeof(alg))
getADtype(::Type{<: VariationalInference{AD}}) where {AD} = AD

abstract type VariationalObjective end

const VariationalPosterior = Distribution{Multivariate, Continuous}

"""
    rand(vi::VariationalInference, num_samples)

Produces `num_samples` samples for the given VI method using number of samples equal to `num_samples`.
"""
function rand(vi::VariationalPosterior, num_samples) end

"""
    objective(vi::VariationalInference, num_samples)

Computes empirical estimates of ELBO for the given VI method using number of samples equal to `num_samples`.
"""
function objective(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model, num_samples) end

# (::VariationalObjective)(vi::VariationalInference, model::Model, num_samples, args...; kwargs...) = begin
# end

"""
    optimize(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model)

Finds parameters which maximizes the ELBO for the given VI method.
"""
function optimize(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model) end

"""
    grad(vo::VariationalObjective, vi::VariationalInference)

Computes the gradients used in `optimize`.

Variance reduction techniques, e.g. control variates, should be implemented in this function.
"""
function grad(vo::VariationalObjective, vi::VariationalInference, q::VariationalPosterior, model::Model) end

"""
    vi(model::Model, alg::VariationalInference)
    vi(model::Model, alg::VariationalInference, q::VariationalPosterior)

Constructs the variational posterior from the `model` using ``
"""
function vi(model::Model, alg::VariationalInference) end

# distributions
include("distributions.jl")

# objectives
include("objectives.jl")

# VI algorithms
include("advi.jl")
include("bbvi.jl")

end
