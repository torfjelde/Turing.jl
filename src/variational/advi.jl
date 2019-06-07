"""
    ADVI(samplers_per_step = 10, max_iters = 5000)

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
struct ADVI{AD} <: VariationalInference{AD}
    samples_per_step # number of samples used to estimate the ELBO in each optimization step
    max_iters        # maximum number of gradient steps used in optimization
end

ADVI(args...) = ADVI{ADBackend()}(args...)
ADVI() = ADVI(10, 5000)

alg_str(::ADVI) = "ADVI"

vi(model::Model, alg::ADVI; optimizer = ADAGrad()) = begin
    # setup
    var_info = VarInfo()
    model(var_info, SampleFromUniform())
    num_params = size(var_info.vals, 1)

    dists = var_info.dists
    ranges = var_info.ranges
    
    q = MeanFieldTransformed(zeros(num_params), zeros(num_params), dists, ranges)

    # construct objective
    elbo = ELBO()

    Turing.DEBUG && @debug "Optimizing ADVI..."
    θ = optimize(elbo, alg, q, model; optimizer = optimizer)
    μ, ω = θ[1:length(q)], θ[length(q) + 1:end]

    # TODO: make mutable instead?
    MeanFieldTransformed(μ, ω, dists, ranges) 
end

# TODO: implement optimize like this?
# (advi::ADVI)(elbo::EBLO, q::MeanFieldTransformed, model::Model) = begin
# end

function optimize(elbo::ELBO, alg::ADVI, q::MeanFieldTransformed, model::Model; optimizer = ADAGrad())
    θ = randn(2 * length(q))
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

function optimize!(elbo::ELBO, alg::ADVI{AD}, q::MeanFieldTransformed, model::Model, θ; optimizer = ADAGrad()) where AD
    alg_name = alg_str(alg)
    samples_per_step = alg.samples_per_step
    max_iters = alg.max_iters
    
    # setup
    # var_info = Turing.VarInfo()
    # model(var_info, Turing.SampleFromUniform())
    # num_params = size(var_info.vals, 1)
    num_params = length(q)

    # # buffer
    # θ = zeros(2 * num_params)

    # HACK: re-use previous gradient `acc` if equal in value
    # Can cause issues if two entries have idenitical values
    if θ ∉ keys(optimizer.acc)
        vs = [v for v ∈ keys(optimizer.acc)]
        idx = findfirst(w -> vcat(q.μ, q.ω) == w, vs)
        if idx != nothing
            @info "[$alg_name] Re-using previous optimizer accumulator"
            θ .= vs[idx]
        end
    else
        @info "[$alg_name] Already present in optimizer acc"
    end
    
    diff_result = DiffResults.GradientResult(θ)

    # TODO: in (Blei et al, 2015) TRUNCATED ADAGrad is suggested; this is not available in Flux.Optimise
    # Maybe consider contributed a truncated ADAGrad to Flux.Optimise

    i = 0
    prog = PROGRESS[] ? ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0) : 0

    time_elapsed = @elapsed while (i < max_iters) # & converged # <= add criterion? A running mean maybe?
        # TODO: separate into a `grad(...)` call; need to manually provide `diff_result` buffers
        # ForwardDiff.gradient!(diff_result, f, x)
        grad!(elbo, alg,q, model, θ, diff_result, samples_per_step)

        # apply update rule
        Δ = DiffResults.gradient(diff_result)
        Δ = Optimise.apply!(optimizer, θ, Δ)
        @. θ = θ - Δ
        
        Turing.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result) norm(DiffResults.gradient(diff_result))
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    @info time_elapsed

    return θ
end

function (elbo::ELBO)(alg::ADVI, q::MeanFieldTransformed, model::Model, θ::AbstractVector{T}, num_samples) where T <: Real
    # setup
    var_info = Turing.VarInfo()

    # initialize `VarInfo` object
    model(var_info, Turing.SampleFromUniform())

    num_params = length(q)
    μ, ω = θ[1:num_params], θ[num_params + 1: end]
    
    elbo_acc = 0.0

    # TODO: instead use `rand(q, num_samples)` and iterate through?

    for i = 1:num_samples
        # iterate through priors, sample and update
        for i = 1:size(q.dists, 1)
            prior = q.dists[i]
            r = q.ranges[i]

            # mean-field params for this set of model params
            μ_i = μ[r]
            ω_i = ω[r]

            # obtain samples from mean-field posterior approximation
            η = randn(length(μ_i))
            ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
            
            # inverse-transform back to domain of original priro
            θ = invlink(prior, ζ)

            # update
            var_info.vals[r] = θ

            # add the log-det-jacobian of inverse transform;
            # `logabsdet` returns `(log(abs(det(M))), sign(det(M)))` so return first entry
            elbo_acc += logabsdet(jac_inv_transform(prior, ζ))[1] / num_samples
        end

        # compute log density
        model(var_info)
        elbo_acc += var_info.logp / num_samples
    end

    # add the term for the entropy of the variational posterior
    variational_posterior_entropy = sum(ω)
    elbo_acc += variational_posterior_entropy

    elbo_acc
end

function (elbo::ELBO)(alg::ADVI, q::MeanFieldTransformed, model::Model, num_samples)
    # extract the mean-field Gaussian params
    θ = vcat(q.μ, q.ω)

    elbo(alg, q, model, θ, num_samples)
end
