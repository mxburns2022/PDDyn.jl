import MathOptInterface as MOI
import SparseArrays as Sp
import Printf
using JuMP
using Ipopt
using LinearAlgebra

include("utilities.jl")
include("rk45.jl")
# import Printf


function to_moi(x::Vector{MOI.VariableIndex}, Q::Matrix{Float64}, c::Vector{Float64}, b::Float64)
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    affine_terms = MOI.ScalarAffineTerm{Float64}[]
    N = size(Q)[1]
    for i in 1:N
        push!(affine_terms, MOI.ScalarAffineTerm(c[i], x[i]))
        push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, i], x[i], x[i]))
        for j in i+1:N
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end 
    return MOI.ScalarQuadraticFunction(quad_terms, affine_terms, b)
end
function get_grad(
        problem::QPBlockData{Float64},
        J::Vector{Float64},
        structure::Vector{Tuple{Int64, Int64}},
        dprimal::Vector{Float64},
        ddual::Vector{Float64},
        primal::Vector{Float64},
        dual::Vector{Float64})
    # First get the primal gradient
    MOI.eval_objective_gradient(problem, dprimal, primal)
    _ = MOI.eval_constraint_jacobian(problem, J, primal)
    for ((i, j), val) in zip(structure, J)
        dprimal[j] += dual[i] * val
    end
    # Now get the objective gradient
    MOI.eval_constraint(problem, ddual, primal);
end


function solve_pddyn(
    problem::QPBlockData{Float64},
    n::Int,
    tol::Float64 = 1e-8,
    verbose::Bool = true,
    log_freq::Int = 1_000,
    τ::Float64 = 1e-3,
    tstop::Float64 = 50.0
)
    printf(x::Float64) = Printf.@sprintf("% 1.6e", x)
    printf(x::Int) = Printf.@sprintf("%6d", x)
    # Implement primal-dual dynamics using the differential equation solver
    m = size(problem.constraints)[1]
    # n = problem.parameters
    Jstructure = MOI.jacobian_structure(problem)
    J = zeros(size(Jstructure));
    # copyto!(primal_dual_system, dual_dual_inds, -P, CartesianIndices(P))
    # copyto!(primal_dual_system, primal_dual_inds, A', CartesianIndices(A'))
    # Projected gradient descent on Lagrangian
    #       L(x, \lambda) = 
    function grad(dx, x, t)
        dprimal = zeros(n)
        ddual = zeros(m)
        get_grad(problem, J, Jstructure, dprimal, ddual, x[1:n], x[n+1:end])
        dx[1:n] .= -dprimal
        dx[n+1:end] .= ddual
    end
    integrator = RK45Integrator(n+m, τ)
    time = 0.0
    x = zeros(n+m)
    n_scratch = zeros(n)
    m_scratch = zeros(m)
    k = 0
    status = MOI.OTHER_ERROR
    while status == MOI.OTHER_ERROR && time < tstop
        
        # Take a single gradient step
        rks_step!(integrator, x, grad, time)
        x[n+1:end] .= max.(0., x[n+1:end])
        if time >= tstop
            status = MOI.ITERATION_LIMIT
        end
        if verbose && (mod(k, log_freq) == 0 || status != MOI.OTHER_ERROR)
            testvec = fill(0.0, m);
            MOI.eval_constraint(problem, testvec, x[1:n])
            comp = testvec .* x[n+1:end]
            logs = printf.((time, MOI.eval_objective(problem, x[1:n]), x[n+1], norm(comp)))
            # println(testvec .* x[n+1:end])
            println(join(logs, "\t"))
        end
        time += τ   
        k += 1
    end
    return status, time, x[1:n], x[n+1:m+n]
end



N = 500
y = MOI.VariableIndex.(1:N)
c_obj = rand(N)
Q_obj = Sp.spdiagm(N, N, ones(N))
model = Model(Ipopt.Optimizer)

@variable(model, x[1:N])
@objective(model, MIN_SENSE, c_obj' * x + 0.5 * x' * Q_obj * x)
c2 = rand(N)
Q2 = c2 * c2'
c3 = rand(N)
b = 0.
f2 = to_moi(y, Q2, c3, b)
@constraint(model, 0.5 * x' * Q2 * x  + c3' * x <= b)
c2 = rand(N)
Q2 = c2 * c2'
c3 = rand(N)
b = 0.
f3 = to_moi(y, Q2, c3, b)
@constraint(model, 0.5 * x' * Q2 * x  + c3' * x <= b)
c2 = rand(N)
Q2 = c2 * c2'
c3 = rand(N)
b = 0.
f4= to_moi(y, Q2, c3, b)
@constraint(model, 0.5 * x' * Q2 * x  + c3' * x <= b)

test = QPBlockData{Float64}()
f1 = MOI.ScalarQuadraticFunction(
    convert(Array{MOI.ScalarQuadraticTerm{Float64}, 1}, [
        MOI.ScalarQuadraticTerm(1, y[i], y[i])
        for i in 1:N
    ]),
    [
        MOI.ScalarAffineTerm(c_obj[i], y[i])
        for i in 1:N
    ],
    0.0
)



# obj = MOI.Objective()
MOI.set(test, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), f1)
MOI.add_constraint(test, f2, MOI.LessThan(b))
MOI.add_constraint(test, f3, MOI.LessThan(b))
MOI.add_constraint(test, f4, MOI.LessThan(b))
optimize!(model)
solve_pddyn(test, N);