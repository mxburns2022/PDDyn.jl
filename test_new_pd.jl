import PowerModels as PM
using Debugger
include("MOI_wrapper.jl")
power_file = ENV["PGLIB"] * "/pglib_opf_case3_lmbd.m";

data = parse_file(power_file)
porig = PowerFlowProblem(data)
model = PM.instantiate_model(data, PM.ACRPowerModel, PM.build_opf)
optimize_model!(model, optimizer=Optimizer)

# p = quantize(porig, 8)
# V = [1.1000000050616032, 0.918748, 0.859438, 4.6790506294074145e-35, 0.117023, -0.267144]
# function rnginit(i)
#     return rand()
# end
# model = Model(Ipopt.Optimizer)
# @variable(model, V[i=1:2p.n], start=rnginit(i))
# power_list = []
# for (j, (Ψⱼ, Φⱼ, Mⱼ, pd, qd, qmax, qmin, pmax, pmin, vmax, vmin)) in enumerate(zip(
#     p.Ψ, p.Φ, p.M, p.p_load, p.q_load, p.q_upper, p.q_lower, p.p_upper, p.p_lower, p.v_mag_max, p.v_mag_min))

#     p_j = V' * Ψⱼ * V
#     print(Ψⱼ)
#     push!(power_list)
#     q_j = V' * Φⱼ * V
#     mag_j = V' * Mⱼ * V
#     @constraint(model, pmin <= (pd + p_j) <= pmax )
#     @constraint(model, qmin <= (qd + q_j) <= qmax )
#     @constraint(model, vmin <= mag_j <= vmax )
# end
# # @constraint(model, V[p.n+p.reference_bus] == 0 )
# @objective(model, MIN_SENSE, V[p.n+1])
# optimize!(model)
# result = solve_pddyn(porig; tstop=50., bits=0)
# ip_result = solve_opf(power_file, PM.ACRPowerModel, Ipopt.Optimizer)
