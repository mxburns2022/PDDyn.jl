include("pdhg.jl")


# MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time
# MOI.get(model::Optimizer, ::MOI.BarrierIterations) = model.iterations
# A = [0.0 -1.0 -1.0 0.0 0.0; 6.0 8.0 0.0 -1.0 0.0; 7.0 12.0 0.0 0.0 -1.0]
# b = [-3.0, 100.0, 120.0]
# c = [12.0, 20.0, 0.0, 0.0, 0.0]
# Q = LinearAlgebra.Diagonal(ones(size(c)))
# Q .*= .1
# Q = c*c'
base_path = ENV["BENCH"] * "/gridopt/pglib-opf"
power_file = base_path * "/pglib_opf_case14_ieee.m"
data = PM.parse_file(power_file)
model =  PM.instantiate_model(data, PM.ACRPowerModel, PM.build_opf).model
