using PowerModels
using Debugger
include("PDDyn.jl")
power_file = ENV["PGLIB"] * "/pglib_opf_case3_lmbd.m";

data = parse_file(power_file)
problem = PowerFlowProblem(data)
V = [1.1000000050616032, 1.0864554011184095, 1.0773745270552761, 4.6790506294074145e-35, 0.17208914221302823, -0.22195526495444665]
