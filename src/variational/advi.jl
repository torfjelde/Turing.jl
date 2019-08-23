using StatsFuns

"""
    ADVI(samples_per_step = 10, max_iters = 5000)

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
struct ADVI{AD} <: VariationalInference{AD}
    samples_per_step # number of samples used to estimate the ELBO in each optimization step
    max_iters        # maximum number of gradient steps used in optimization
end

ADVI(args...) = ADVI{ADBackend()}(args...)
ADVI() = ADVI(10, 5000)

alg_str(::ADVI) = "ADVI"

"""
    meanfield(model::Model)

Creates a mean-field approximation with MvNormal as underlying distribution.
"""
function meanfield(model::Model)
    # setup
    varinfo = Turing.VarInfo(model)
    num_params = sum([size(varinfo.metadata[sym].vals, 1)
                      for sym ∈ keys(varinfo.metadata)])

    dists = vcat([varinfo.metadata[sym].dists for sym ∈ keys(varinfo.metadata)]...)

    num_ranges = sum([length(varinfo.metadata[sym].ranges)
                      for sym ∈ keys(varinfo.metadata)])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1
    for sym ∈ keys(varinfo.metadata)
        for r ∈ varinfo.metadata[sym].ranges
            ranges[range_idx] = idx .+ r
            range_idx += 1
        end
        
        # append!(ranges, [idx .+ r for r ∈ varinfo.metadata[sym].ranges])
        idx += varinfo.metadata[sym].ranges[end][end]
    end

    # construct variatioanl posterior
    μ = randn(num_params)
    σ = softplus.(randn(num_params))
    # σ = Distributions.rand(InverseGamma(3, 1), num_params)

    d = TuringDiagNormal(μ, σ)
    bs = inv.(bijector.(tuple(dists...)))
    b = Stacked(bs, ranges)

    return transformed(d, b)
end


function vi(model::Model, alg::ADVI; optimizer = TruncatedADAGrad())
    q = meanfield(model)
    return vi(model, alg, q; optimizer = optimizer)
end

# TODO: make more flexible, allowing other types of `q`
function vi(
    model::Model,
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal};
    optimizer = TruncatedADAGrad()
)
    Turing.DEBUG && @debug "Optimizing ADVI..."
    θ = optimize(elbo, alg, q, model; optimizer = optimizer)
    μ, ω = θ[1:length(q)], θ[length(q) + 1:end]

    return update(q, μ, softplus.(ω))
end

function optimize(
    elbo::ELBO,
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal},
    model::Model;
    optimizer = TruncatedADAGrad()
)
    μ, σs = params(q)
    θ = vcat(μ, log.(σs))

    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

function (elbo::ELBO)(
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal},
    model::Model,
    θ::AbstractVector{T},
    num_samples
) where T <: Real
    # setup
    varinfo = Turing.VarInfo(model)
    
    num_params = length(q)
    μ = @view θ[1:num_params]
    ω = @view θ[num_params + 1: end]

    # update the variational posterior
    q = update(q, μ, softplus.(ω))

    # sample from variational posterior
    samples = Distributions.rand(q, num_samples)

    elbo_acc = 0.0
    
    for i = 1:num_samples
        z = samples[:, i]

        # results in an issue if `z`
        varinfo = VarInfo(varinfo, SampleFromUniform(), z)
        model(varinfo)

        # `logabsdetjac` here is actually `logabsdetjacinv`
        elbo_acc += (varinfo.logp + logabsdetjac(q, z)) / num_samples
    end

    elbo_acc -= entropy(q)

    return elbo_acc
end

function (elbo::ELBO)(
    alg::ADVI,
    q::TransformedDistribution{<: TuringDiagNormal},
    model::Model,
    num_samples
)
    # extract the mean-field Gaussian params
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    return elbo(alg, q, model, θ, num_samples)
end
