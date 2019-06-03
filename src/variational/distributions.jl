import Distributions: _rand!, _logpdf
using Distributions

function jac_inv_transform(dist::Distribution, x::T where T<:Real)
    ForwardDiff.derivative(x -> invlink(dist, x), x)
end

function jac_inv_transform(dist::Distribution, x::AbstractArray{T} where T <: Real)
    ForwardDiff.jacobian(x -> invlink(dist, x), x)
end

function jac_inv_transform(dist::Distribution, x::TrackedArray{T} where T <: Real)
    Tracker.jacobian(x -> invlink(dist, x), x)
end

function center_diag_gaussian(x, μ, σ)
    # instead of creating a diagonal matrix, we just do elementwise multiplication
    (σ .^(-1)) .* (x - μ)
end

function center_diag_gaussian_inv(η, μ, σ)
    (η .* σ) + μ
end

# TODO: Make `MeanField` work with arbitrary list of distributions, i.e. not assuming everything Gaussian
# Mean-field without transformation
struct MeanField <: VariationalPosterior
    μ
    ω
end

MeanField(θ) = begin
    @assert length(θ) % 2 == 0
    mid = Integer(length(θ) / 2)
    MeanField(θ[1:mid], θ[mid + 1:end])
end

Base.length(q::MeanField) = length(q.μ)
params(q::MeanField) = vcat(q.μ, q.ω)

_rand!(rng::AbstractRNG, q::MeanField, x::AbstractVector{T}) where {T <: Real} = begin
    y = randn(rng, length(q))
    z = center_diag_gaussian_inv.(y, q.μ, exp.(q.ω))
    
    return z
end

_logpdf(q::MeanField, x::AbstractVector{T}) where {T <: Real} = begin
    sum([logpdf(Normal(q.μ[i], exp(q.ω[i])), x[i]) for i = 1:length(x)])
end

# Mean-field approximation used by ADVI
struct MeanFieldTransformed{TDists <: AbstractVector{<: Distribution}} <: VariationalPosterior
    μ
    ω
    dists::TDists
    ranges::Vector{UnitRange{Int}}
end

Base.length(q::MeanFieldTransformed) = length(q.μ)
params(q::MeanFieldTransformed) = vcat(q.μ, q.ω)

_logpdf(q::MeanFieldTransformed, x::AbstractVector{T}) where {T <: Real} = begin
    sum([logpdf(Normal(q.μ[i], exp(q.ω[i])), x[i]) for i = 1:length(x)])
end

_rand!(rng::AbstractRNG, q::MeanFieldTransformed{TDists}, x::AbstractVector{T}) where {T<:Real, TDists <: AbstractVector{<: Distribution}} = begin
    # extract parameters for convenience
    μ, ω = q.μ, q.ω
    num_params = length(q)

    for i = 1:size(q.dists, 1)
        prior = q.dists[i]
        r = q.ranges[i]

        # initials
        μ_i = μ[r]
        ω_i = ω[r]

        # # sample from VI posterior
        η = randn(rng, length(μ_i))
        ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
        θ = invlink(prior, ζ)

        x[r] = θ
    end

    return x
end

# indexing interface
Base.getindex(q::MeanFieldTransformed, i) = Normal(q.μ[i], exp(q.ω[i]))
Base.firstindex(q::MeanFieldTransformed) = 1
Base.lastindex(q::MeanFieldTransformed) = length(q)
