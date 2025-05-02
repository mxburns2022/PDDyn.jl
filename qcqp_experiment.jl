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

function run_pddyn(m::Model; bits::Int = 0, delay::Float64=0.0, randomize::Bool = false)
    block, lbounds, ubounds, N = convert_to_format(m.moi_backend)
    new_block = quantize(block, bits, randomize)
    # println(new_block.objective)
    status, primal_status, dual_status, time, objective, primal, dual = solve_pddyn(new_block, N, lbounds, ubounds; tstop = 10., verbose=false, τ = 3e-5, tol=1e-4, target=block, delay=delay)
    # objective = 
    return time, objective, primal_status, dual_status, primal
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
            :dual_status => [],
            :bits => [],
            :delay => [],
            :distance => [],
            :random => [])
# m = Model()
# @variable(m, x[1:2])
# # @constraint(m, x[1] )
# @constraint(m, x[1]^2 <= 2)
# @constraint(m, x[1]^2 >= 0)
# @constraint(m, x[2]^2+x[1]^2 <= 2)
# @constraint(m, x[2]^2 >= 0)
# # @constraint(m, -x[2] + x[1] <= 0.5)
# @objective(m, Min, x[1]^2 + x[2]^2+x[2])
# time, objective, primal_status, dual_status, primal = run_pddyn(m)

bits=10
for path in readdir("instances/qcqp")
    if !contains(path, "c_30") && !contains(path, "c_10")
        continue
    end 
    m = fetch_model_path("/home/matt/Documents/development/optimization/PDDyn.jl/instances/qcqp/$(path)")
    set_optimizer(m, Ipopt.Optimizer)
    set_silent(m)
    optimize!(m)
    ipopt_obj = objective_value(m)
    x_actual = value.(m[:x]).data
    for delay in [1e-3, 1e-2, 1e-1, 2e-1, 4e-1]
        for random in [false]
            time, objective, primal_status, dual_status, x = run_pddyn(m;bits=16, delay=delay, randomize=random)
            println("$(path), $(delay), $(random), $(objective)")
            push!(data[:benchmark], path)
            push!(data[:time], time)
            push!(data[:objective], objective)
            push!(data[:ipopt_objective], objective_value(m))
            push!(data[:primal_status], (primal_status == MOI.FEASIBLE_POINT))
            push!(data[:dual_status], (dual_status == MOI.FEASIBLE_POINT))
            push!(data[:bits], bits)
            push!(data[:distance], norm(x_actual[2:end] - x[2:end]))
            push!(data[:delay], delay)
            push!(data[:random], random)
        end
        # for random in [true]
        #     for i in 1:20
        #         time, objective, primal_status, dual_status, x = run_pddyn(m;bits=16, delay=delay, randomize=random )
        #         println("$(path), $(bits), $(random), $(objective)")
        #         push!(data[:benchmark], path)
        #         push!(data[:time], time)
        #         push!(data[:objective], objective)
        #         push!(data[:ipopt_objective], objective_value(m))
        #         push!(data[:primal_status], (primal_status == MOI.FEASIBLE_POINT))
        #         push!(data[:dual_status], (dual_status == MOI.FEASIBLE_POINT))
        #         push!(data[:bits], bits)
        #         push!(data[:distance], norm(x_actual[2:end] - x[2:end]))
        #         push!(data[:random], random)
        #         push!(data[:delay], delay)
        #     end
        # end
    end
#    break
    df = DataFrame(data)
    CSV.write("qcqp1_experiment_delay_30.csv", df)
end
# # # # m2 = fetch_model_path("/home/matt/Documents/development/optimization/PDDyn.jl/instances/qcqp/unitbox_c_8_8_3_25.jl")
# # block, lbounds, ubounds, N = convert_to_format(m.moi_backend)
# # ----- Objective ----- #
# # @objective(m, Min, x[1])

 

# # set_optimizer(m2, Ipopt.Optimizer)
# # optimize!(m2)
# # # println(value.(x))
# # set_optimizer(m, Optimizer)
# # optimize!(m)
# # println(objective_value(m), objective_value(m2))


