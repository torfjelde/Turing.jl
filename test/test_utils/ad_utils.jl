using Turing: gradient_logp_forward, gradient_logp_reverse
using Test
using Zygote: Zygote
using Tracker: Tracker
using ForwardDiff: ForwardDiff
using FDM
using Turing.Core.RandomVariables: getval

function test_ad(f, at = 0.5; rtol = 1e-8, atol = 1e-8)
    isarr = isa(at, AbstractArray)
    tracker = Tracker.gradient(f, at)[1]
    zygote = Zygote.gradient(f, at)[1]
    @test isapprox(tracker, zygote, rtol=rtol, atol=atol)
    if isarr
        forward = ForwardDiff.gradient(f, at)
        @test isapprox(tracker, forward, rtol=rtol, atol=atol)
    else
        forward = ForwardDiff.derivative(f, at)
        finite_diff = central_fdm(5,1)(f, at)
        @test isapprox(tracker, forward, rtol=rtol, atol=atol)
        @test isapprox(tracker, finite_diff, rtol=rtol, atol=atol)
    end
end

function test_model_ad(model, f, syms::Vector{Symbol})
    # Set up VI.
    vi = Turing.VarInfo()
    model(vi, SampleFromPrior())

    # Collect symbols.
    vals = Float64[]
    for i in 1:length(syms)
        s = syms[i]
        vns = filter(vn -> vn.sym == s, vi.vns)
        vals = [vals; getval(vi, vns)] 
    end

    spl = SampleFromPrior()
    _, ∇E = gradient_logp_forward(vi[spl], vi, model)
    grad_Turing = sort(∇E)

    # Call ForwardDiff's AD
    grad_FWAD = sort(ForwardDiff.gradient(f, vec(vals)))

    # Compare result
    @test grad_Turing ≈ grad_FWAD atol=1e-9
end
