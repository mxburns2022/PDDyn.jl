using JuMP
using Ipopt

N = 10
model = Model(Ipopt.Optimizer)
@variable(model, Vr >= 0)
@variable(model, Vi >= 0)

@constraint(model, powerlow <= Vr .* Vi <= powerhigh)

powerlow = rand(N) * 2 - 1
powerhigh = rand(N) * 2 + 1
cost = rand(N) * 10 + 10
@constraint(model, powerlow <= V .* V <= powerhigh)
@constraint(model)