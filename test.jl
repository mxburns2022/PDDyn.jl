include("MOI_wrapper.jl")
using MINLPLib
using DataFrames
import CSV
using JuMP
function fetch_model_path(instance::AbstractString;options::Dict=Dict())

    if isfile(instance)
        m = include(instance)
    else
        @warn "No instances detected..."
        return nothing
    end
    
    if typeof(m) == JuMP.Model
        return m
    else
        return m.model
    end
end

function run_pddyn(m::Model)
    block, lbounds, ubounds, N = convert_to_format(m.moi_backend)
    status, primal_status, dual_status, time, objective, primal, dual = solve_pddyn(block, N, lbounds, ubounds; tstop = 100., verbose=false, τ = 1e-5, tol=1e-4)
    return time, objective, primal_status, dual_status
end
# ----- Objective ----- #
# @objective(m, Min, x[1])

 

# ----- Objective ----- #
# @objective(m, Min, x[1])

data = Dict(:benchmark => [],
            :time => [],
            :objective => [],
            :ipopt_objective => [],
            :primal_status => [],
            :dual_status => [])
for path in readdir("instances/qcqp")
    m = fetch_model_path("/home/matt/Documents/development/optimization/PDDyn.jl/instances/qcqp/$(path)")
    println(path)
    set_optimizer(m, Ipopt.Optimizer)
    set_silent(m)
    optimize!(m)
    time, objective, primal_status, dual_status = run_pddyn(m)
    push!(data[:benchmark], path)
    push!(data[:time], time)
    push!(data[:objective], objective)
    push!(data[:ipopt_objective], objective_value(m))
    push!(data[:primal_status], (primal_status == MOI.FEASIBLE_POINT))
    push!(data[:dual_status], (dual_status == MOI.FEASIBLE_POINT))
    df = DataFrame(data)
    CSV.write("qcqp1_experiment.csv", df)
end
# m2 = fetch_model_path("/home/matt/Documents/development/optimization/PDDyn.jl/instances/qcqp/unitbox_c_8_8_3_25.jl")
# block, lbounds, ubounds, N = convert_to_format(m.moi_backend)
# ----- Objective ----- #
# @objective(m, Min, x[1])

 

# set_optimizer(m2, Ipopt.Optimizer)
# optimize!(m2)
# # println(value.(x))
# set_optimizer(m, Optimizer)
# optimize!(m)
# println(objective_value(m), objective_value(m2))


