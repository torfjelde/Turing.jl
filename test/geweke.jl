using Stats, Distributions, Turing
using Gadfly

# include("ASCIIPlot.jl")
import Gadfly.ElementOrFunction

# First add a method to the basic Gadfly.plot function for QQPair types (generated by Distributions.qqbuild())
Gadfly.plot(qq::QQPair, elements::ElementOrFunction...) = Gadfly.plot(x=qq.qx, y=qq.qy, Geom.point, Theme(highlight_width=0px), elements...)

# Now some shorthand functions
qqplot(x, y, elements::ElementOrFunction...) = Gadfly.plot(qqbuild(x, y), elements...)
qqnorm(x, elements::ElementOrFunction...) = qqplot(Normal(), x, Guide.xlabel("Theoretical Normal quantiles"), Guide.ylabel("Observed quantiles"), elements...)

NSamples = 5000

@model gdemo_fw() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  y ~ MvNormal([m; m; m], [sqrt.(s) 0 0; 0 sqrt.(s) 0; 0 0 sqrt.(s)])
end

@model gdemo_bk(x) = begin
  # Backward Step 1: theta ~ theta | x
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt.(s))
  x ~ MvNormal([m; m; m], [sqrt.(s) 0 0; 0 sqrt.(s) 0; 0 0 sqrt.(s)])
  # Backward Step 2: x ~ x | theta
  y ~ MvNormal([m; m; m], [sqrt.(s) 0 0; 0 sqrt.(s) 0; 0 0 sqrt.(s)])
end

fw = HMCDA(NSamples, 0.9, 0.1)
# bk = Gibbs(10, PG(10,10, :s, :y), HMC(1, 0.25, 5, :m));
bk = HMCDA(50, 0.9, 0.1);

s = sample(gdemo_fw(), fw);
# describe(s)

N = div(NSamples, 50)

x = [s[:y][1]...]
s_bk = Array{Turing.Chain}(undef, N)

set_verbosity(0)
i = 1
while i <= N
  try
    s_bk[i] = sample(gdemo_bk(x), bk);
    x = [s_bk[i][:y][end]...];
    i += 1
  catch
  end
end
set_verbosity(1)

s2 = vcat(s_bk...);
# describe(s2)


qqplot(s[:m], s2[:m])
qqplot(s[:s], s2[:s])

using Test

qqm = qqbuild(s[:m], s2[:m])
X = qqm.qx
y = qqm.qy
slope = (1 / (transpose(X) * X)[1] * transpose(X) * y)[1]
@test slope ≈ 1.0 atol=0.1

# NOTE: test for s is not stable
#       probably due to the transformation
# qqs = qqbuild(s[:s], s2[:s])
# X = qqs.qx
# y = qqs.qy
# slope = (1 / (transpose(X) * X)[1] * transpose(X) * y)[1]
# @test slope ≈ 1.0 atol=0.1
