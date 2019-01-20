##############
# Utilities  #
##############

# VarInfo to Sample
function Sample(vi::AbstractVarInfo, weight = 0.0, init_zero = false)
    # NOTE: do we need to check if lp is 0?
    lp = init_zero ? 0.0 : getlogp(vi)
    lf_eps, elapsed, epsilon = NaN, NaN, NaN
    lf_num, eval_num = 0, 0

    Sample(weight, lp, lf_eps, elapsed, epsilon, lf_num, eval_num, vi)
end

# VarInfo, combined with spl.info, to Sample
function Sample(vi::AbstractVarInfo, spl::Sampler)
    s = Sample(vi)

    if haskey(spl.info, :wum)
        s.epsilon = getss(spl.info[:wum])
    end

    if haskey(spl.info, :lf_num)
        s.lf_num = spl.info[:lf_num]
    end

    if haskey(spl.info, :eval_num)
        s.eval_num = spl.info[:eval_num]
    end

    return s
end

function sample(model, alg; kwargs...)
    spl = get_sampler(model, alg; kwargs...)
    vi = init_varinfo(model, spl; kwargs...)
    samples = init_samples(alg, vi; kwargs...)
    _sample(vi, samples, spl, model, alg; kwargs...)
end

function init_samples(alg, vi::AbstractVarInfo; kwargs...)
    n = get_sample_n(alg; kwargs...)
    return [Sample(vi, 1/n, true) for i in 1:n]
end

function get_sample_n(alg; reuse_spl_n = 0, kwargs...)
    if reuse_spl_n > 0
        return reuse_spl_n
    else
        alg.n_iters
    end
end

function get_sampler(model, alg; kwargs...)
    spl = default_sampler(model, alg; kwargs...)
    if alg isa AbstractGibbs
        @assert typeof(spl.alg) == typeof(alg) "[Turing] alg type mismatch; please use resume() to re-use spl"
    end
    return spl
end

function default_sampler(model, alg; reuse_spl_n = 0, resume_from = nothing, kwargs...)
    if reuse_spl_n > 0
        return resume_from.info[:spl]
    else
        return Sampler(alg, model)
    end
end

function init_varinfo(model, spl; kwargs...)
    vi = TypedVarInfo(default_varinfo(model, spl; kwargs...))
    return vi
end

function default_varinfo(model, spl; resume_from = nothing, kwargs...)
    if resume_from == nothing
        vi = VarInfo()
        model(vi, HamiltonianRobustInit())
        return vi
    else
        return resume_from.info[:vi]
    end
end
