struct BBVI{AD}  <: VariationalInference{AD}
    samples_per_step::Integer
    max_iters::Integer
end

BBVI(args...) = BBVI{ADBackend()}(args...)
BBVI() = BBVI(10, 1000)

alg_str(::BBVI) = "BBVI"

vi(model::Model, alg::BBVI, q::MeanFieldTransformed; optimizer = ADAGrad()) where D <: VariationalPosterior = begin
    alg_name = alg_str(alg)
    elbo = ELBO()

    Turing.DEBUG && @debug "Optimizing $alg_name..."
    θ = optimize(elbo, alg, q, model; optimizer = optimizer)

    mid = Integer(length(θ) / 2)

    return MeanFieldTransformed(θ[1:mid], θ[mid + 1:end], q.dists, q.ranges)
end

function optimize(elbo::ELBO, alg::BBVI, q, model::Model; optimizer = ADAGrad())
    θ = randn(length(vcat(params(q)...)))
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

function optimize!(elbo::ELBO, alg::BBVI{AD}, q, model::Model, θ; optimizer = AdaGrad()) where AD
    alg_name = alg_str(alg)
    max_iters = alg.max_iters

    diff_result = DiffResults.GradientResult(θ)

    i = 0
    prog = PROGRESS[] ? ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0) : 0

    time_elapsed = @elapsed while (i < max_iters) # & converged # <= add criterion? A running mean maybe?
        # TODO: separate into a `grad(...)` call; need to manually provide `diff_result` buffers
        # ForwardDiff.gradient!(diff_result, f, x)
        Δ = zeros(length(θ))

        grad!(elbo, alg, q, model, θ, diff_result; samples_per_step = alg.samples_per_step)

        # for i = 1:alg.samples_per_step
        #     grad!(elbo, alg, q, model, θ, diff_result)

        #     # apply update rule
        #     Δ += DiffResults.gradient(diff_result)
        # end
        
        Δ = Optimise.apply!(optimizer, θ, Δ)
        @. θ = θ - Δ
        
        Turing.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result) norm(DiffResults.gradient(diff_result))
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    @info time_elapsed

    return θ
end

function grad!(vo::ELBO, alg::BBVI{AD}, q::MeanFieldTransformed, model::Model, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult, args...; samples_per_step = 10) where {T <: Real, AD <: ForwardDiffAD}
    # TODO: implement variance reduction techniques
    
    # # sample from variational posterior with updated params
    # z = rand(q, S)
    mid = Integer(length(θ) / 2)

    ∇ = zeros(length(θ))
    
    chunk_size = getchunksize(alg)
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))

    for s = 1:samples_per_step
        z = Distributions.rand(q)
        
        # TODO: this probably slows down executation quite a bit; exists a better way of doing this?
        logpdf_(θ_) = begin
            q_ = MeanFieldTransformed(θ_[1:mid], θ_[mid + 1:end], q.dists, q.ranges)
            logpdf(q_, z)
        end

        config = ForwardDiff.GradientConfig(logpdf_, θ, chunk)
        ForwardDiff.gradient!(out, logpdf_, θ)

        ∇ .+= DiffResults.gradient(out) .* (logdensity(model, z) - DiffResults.value(out)) / samples_per_step
    end

    # zs = Distributions.rand(q, samples_per_step)

    # params_per_comp_dist = 2

    # # completely overwritten on each loop, so initialize outside of
    # # loop to reuse buffer instead of repeated allocations
    # fs = zeros(samples_per_step, params_per_comp_dist)  # size s.t. can hold (μᵢ, ωᵢ)
    # hs = zeros(samples_per_step, params_per_comp_dist)
    
    # for i = 1:length(q)        
    #     for s = 1:samples_per_step
    #         z = zs[:, s]

    #         # TODO: generalize to MF approx for arbitrary distributions
    #         logpdf_(θ_) = begin
    #             qᵢ = Normal(θ_[1], exp(θ_[2]))  # construct the MF-distribution
    #             logpdf(qᵢ, z[i])
    #         end

    #         θᵢ = [θ[i], θ[i + length(q)]]  # extract [μ, ω]

    #         config = ForwardDiff.GradientConfig(logpdf_, θᵢ, chunk)
    #         ∇ᵢ = ForwardDiff.gradient(logpdf_, θᵢ)

    #         # TODO: this is wrong; need to do component-wise?
    #         fs[s, :] = ∇ᵢ * (logdensity(model, z) - logpdf(q[i], z[i]))
    #         hs[s, :] = ∇ᵢ
    #     end

    #     # compute optimal scaling factor
    #     a = sum([cov(fs[:, d], hs[:, d]) for d = 1:params_per_comp_dist]) / sum([var(hs[:, d]) for d = 1:params_per_comp_dist])

    #     # store
    #     ∇[i] += mean(fs[:, 1] - a .* hs[:, 1])             # μᵢ
    #     ∇[i + length(q)] += mean(fs[:, 2] - a .* hs[:, 2]) # ωᵢ
    # end

    DiffResults.gradient!(out, - ∇)  # `optimize!` steps in negative direction
end

function (elbo::ELBO)(alg::BBVI, q::MeanFieldTransformed, model::Model, num_samples::Int64)
    # TODO
    zs = Distributions.rand(q, num_samples)
    res = 0.0

    for i = 1:num_samples
        z = zs[:, i]
        res += logdensity(model, z) / num_samples
        res -= logpdf(q, z) / num_samples
    end

    return res
end

logdensity(model::Model, z) = begin
    # setup
    var_info = Turing.VarInfo()

    # initialize `VarInfo` object
    model(var_info, Turing.SampleFromUniform())

    for r ∈ var_info.ranges
        var_info.vals[r] = z[r]
    end

    # compute logdensity
    model(var_info)

    return var_info.logp
end
