import PowerModels as PM
using Debugger
import CSV
using DataFrames
include("MOI_wrapper.jl")
power_file = ENV["PGLIB"] * "/pglib_opf_case3_lmbd.m";
# pm 
data = parse_file(power_file)
for (brkey, brdata) in data["branch"]
  data["branch"][brkey]["rate_a"] = 10000
end

p = PowerFlowProblem(data)
model = PM.instantiate_model(data, PM.ACRPowerModel, PM.build_opf)
# optimize_model!(model, optimizer=Optimizer)
N = length(data["bus"])
# p = quantize(porig, 8)
# V = [1.1000000050616032, 0.918748, 0.859438, 4.6790506294074145e-35, 0.117023, -0.267144]
function rnginit(i)
    return rand()
end
# model = Model(Ipopt.Optimizer)
# @variable(model, V[i=1:2p.n], start=rnginit(i))
result = optimize_model!(model, optimizer=Ipopt.Optimizer)
for b in [0, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  for rd in [true, false]
    p_bits = quantize(p, b, rd)
    pd_vec = solve_pddyn(p_bits;tstop=50000.)
    results_pd = pd_vec[end]
    complementarity = [maximum(u[2N+1:end-p.nbr].*eval_constraints(p, u[1:2N])[1:end-p.nbr]) for u in results_pd.u]
    constraint_vals = [maximum(eval_constraints(p, u[1:2N])) for u in results_pd.u]
    fvals = [eval_objective(p, u[1:2N]) for u in results_pd.u]
    df = DataFrame(
      merge(
        Dict(
          "t" => results_pd.t,
          "objective" => [eval_objective(p, u[1:2N]) for u in results_pd.u],
          "primal_feasibility" => [maximum(eval_constraints(p, u[1:2N])) for u in results_pd.u],
          "dual_feasibility" => [norm(u[2N+1:end].*eval_constraints(p, u[1:2N])) for u in results_pd.u],
        ),
        Dict(
          ["v$(i)" => [u[i] for u in results_pd.u] for i in 1:2N]...
        )
      )
    )
    CSV.write("case_$(N)_bits_$(b)_$(rd)_results.csv", df)
  end
end
# # V = results_pd[1]
# # Pb = []
# # for Φ_i in p.Ψ
# #   push!(Pb, V'*Φ_i*V)
# # end
# # for Φ_i in p.Φ
# #   push!(Pb, V'*Φ_i*V)
# # end
# # for (Ψ_i, Φ_i) in zip(p.Ψbr, p.Φbr)
# #   push!(Pb, (V'*Φ_i*V)^2+(V'*Ψ_i*V)^2)
# # end
# V = vcat([result["solution"]["bus"]["$(i)"]["vr"] for i in 1:3], [result["solution"]["bus"]["$(i)"]["vi"] for i in 1:N])

# for (i, (b, bdict)) in enumerate(result["solution"]["branch"])
#     bval = data["branch"][string( i)]
#     src = convert(Int, bval["f_bus"])
#     dst = convert(Int, bval["t_bus"])
#     pflow_from = bdict["pf"]
#     pflow_to = bdict["pt"]
#     P[src] += pflow_from
#     P[dst] += pflow_to
#     qflow_from = bdict["qf"]
#     qflow_to = bdict["qt"]
#     P[src+N] += qflow_from
#     P[dst+N] += qflow_to
# end
# pd_results = solve_pddyn(p; tstop=7000000., bits=0)
# V = pd_results[1]
# Pb = []

# # ip_result = solve_opf(power_file, PM.ACRPowerModel, Ipopt.Optimizer)
