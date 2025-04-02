import MathOptInterface as MOI
import SparseArrays as sp
import Printf
using JuMP
using Ipopt
using LinearAlgebra
using Random
using PowerModels
using DifferentialEquations
rng = MersenneTwister(123)

include("utilities.jl")
include("rk45.jl")
# import Printf


function quantize(x::Float64, step::Float64)
    return round(x / step) * step
end

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

function to_quantized_moi(x::Vector{MOI.VariableIndex}, Q::Matrix{Float64}, c::Vector{Float64}, b::Float64, bits::Int, maxcoeff::Float64)
    step = maxcoeff / (2 ^ bits)
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    affine_terms = MOI.ScalarAffineTerm{Float64}[]
    cq = quantize.(c, step)
    Qq = quantize.(Q, step)
    bq = quantize(b, step)
    N = size(Q)[1]
    for i in 1:N
        push!(affine_terms, MOI.ScalarAffineTerm(cq[i], x[i]))
        push!(quad_terms, MOI.ScalarQuadraticTerm(Qq[i, i], x[i], x[i]))
        for j in i+1:N
            push!(quad_terms, MOI.ScalarQuadraticTerm(Qq[i, j], x[i], x[j]))
        end
    end 
    return MOI.ScalarQuadraticFunction(quad_terms, affine_terms, bq)
end

function get_grad(
        problem::QPBlockData{Float64},
        J::Vector{Float64},
        structure::Vector{Tuple{Int64, Int64}},
        dprimal::Vector,
        ddual::Vector,
        primal,
        dual)
    # First get the primal gradient
    MOI.eval_objective_gradient(problem, dprimal, primal)
    _ = MOI.eval_constraint_jacobian(problem, J, primal)
    for ((i, j), val) in zip(structure, J)
        dprimal[j] += ForwardDiff.value(dual[i] * val)
    end
    # Now get the objective gradient
    MOI.eval_constraint(problem, ddual, primal);
    # ddual ./= problem.norm_constant
    # dprimal ./= problem.norm_constant
end
function get_proximal_primal_grad(
        problem::QPBlockData{Float64},
        J::Vector{Float64},
        structure::Vector{Tuple{Int64, Int64}},
        dprimal::Vector{Float64},
        ddual::Vector{Float64},
        primal::Vector{Float64},
        previous::Vector{Float64},
        reg::Float64,
        dual::Vector{Float64})
    # First get the primal gradient
    MOI.eval_objective_gradient(problem, dprimal, primal)
    _ = MOI.eval_constraint_jacobian(problem, J, primal)
    for ((i, j), val) in zip(structure, J)
        dprimal[j] += dual[i] * val
    end
    dprimal .+= primal - (previous .* reg)

    # Now get the objective gradient
    MOI.eval_constraint(problem, ddual, primal);
end
function get_dual_grad(
    problem::QPBlockData{Float64},
    ddual::Vector{Float64},
    primal::Vector{Float64})
    # Now get the objective gradient
    MOI.eval_constraint(problem, ddual, primal);
end

function solve_pddyn(
    problem::QPBlockData{Float64},
    n::Int,
    lbounds::Vector{Float64},
    ubounds::Vector{Float64};
    tstop::Float64 = 5000.0,
    τ::Float64 = 1e-3,
    tol::Float64 = 1e-5,
    verbose::Bool = true,
    log_freq::Int = 10_000,
    post_sample::Bool = false,
    target::Union{QPBlockData{Float64}, Type{Nothing}} = Nothing
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
    function grad(dx, x, p, t)
        dprimal = zeros(n)
        ddual = zeros(m)
        get_grad(problem, J, Jstructure, dprimal, ddual, x[1:n], x[n+1:end])
        dx[1:n] .= -dprimal
        dx[n+1:end] .= ddual
        x.= min.(max.(x, lbounds), ubounds)
    end
    time = 0.0
    x = rand(n+m)
    for i in 1:100 
        τ_stop = tstop / 100
        odeproblem = ODEProblem(grad, x, (0.0, τ_stop))
        result = solve(odeproblem,Rodas5P(), save_everystep = false);
        x = result.u[end]
        testvec = fill(0.0, m);
        MOI.eval_constraint(problem, testvec, x[1:n])
        comp = testvec .* x[n+1:end]
        # print(testvec)
        logs = printf.((time, MOI.eval_objective(problem, x[1:n]), max(testvec...), norm(comp)))
        # println(testvec .* x[n+1:end])
        println(join(logs, "\t"))
    end
    k = 0
    status = MOI.OTHER_ERROR
    # while status == MOI.OTHER_ERROR && time < tstop
    #     # Take a single gradient step
    #     rks_step!(integrator, x, grad, time)
    #     x.= min.(max.(x, lbounds), ubounds)
    #     # x[1:n] .= max.(x[1:n], 0.0)
    #     # x[1:n] .= min.(x[1:n], ubounds)
    #     if time >= tstop
    #         status = MOI.ITERATION_LIMIT
    #     end
    #     if verbose && (mod(k, log_freq) == 0 || status != MOI.OTHER_ERROR)
    #         if m > 0
    #             testvec = fill(0.0, m);
    #             MOI.eval_constraint(problem, testvec, x[1:n])
    #             comp = testvec .* x[n+1:end]
    #             # print(testvec)
    #             logs = printf.((time, MOI.eval_objective(problem, x[1:n]), max(testvec...), norm(comp)))
    #             # println(testvec .* x[n+1:end])
    #             println(join(logs, "\t"))
    #         else
    #             logs = printf.((time, MOI.eval_objective(problem, x[1:n])))
    #             # println(testvec .* x[n+1:end])
    #             println(join(logs, "\t"))
    #         end
    #     end
    #     time += τ   
    #     k += 1
    # end
    if post_sample
        grad_buff = zeros(n+m)
        scratch_vals = zeros(m)
        n_iter = 50000
        best_obj = MOI.eval_objective(target, x[1:n])
        bestx = zeros(n+m)
        copy!(bestx, x)
        for i in 1:n_iter
            grad(grad_buff, x, Nothing, time)
            grad_buff[1:n] .+= 1 / 256 * (sqrt( 1 / 1e-4)) * randn(n)
            x += 1e-4 * grad_buff;
            x[n+1:end] .= max.(0., x[n+1:end])
            if time >= tstop
                status = MOI.ITERATION_LIMIT
            end
            if verbose && (mod(i, log_freq) == 0 || status != MOI.OTHER_ERROR)
                obj = MOI.eval_objective(target, x[1:n])
                maxvio = max(scratch_vals...)
                MOI.eval_constraint(target, scratch_vals, x[1:n])
                if obj < best_obj && maxvio < 1e-6
                    
                    println(obj, best_obj);
                    copy!(bestx, x)
                    best_obj = obj
                    println(best_obj);
                end
            end
            time += τ   
        end
        copy!(x, bestx)
    end
    status = MOI.LOCALLY_SOLVED
    constvec = zeros(m)
    MOI.eval_constraint(problem, constvec, x[1:n])
    complementarity = norm(constvec .* x[n+1:m+n]) 
    primal_status = MOI.NO_SOLUTION
    dual_status = MOI.NO_SOLUTION
    if m > 0
        if maximum(constvec) < tol
            primal_status = MOI.FEASIBLE_POINT
        else
            primal_status = MOI.INFEASIBLE_POINT
        end
    else
        primal_status = MOI.FEASIBLE_POINT
    end
    if complementarity < tol
        dual_status = MOI.FEASIBLE_POINT
    end
        
    return status, primal_status, dual_status, time, x[1:n], x[n+1:m+n]
end


function main_test()
    N = 50
    y = MOI.VariableIndex.(1:N)

    c_obj = rand(rng,N)
    # Q_obj = sp.spdiagm(N, N, ones(N))
    Q_obj = diagm(N, N, ones(N))
    maxcoeff = max(maximum(abs.(c_obj)), maximum(abs.(Q_obj)))
    QList = []
    clist = []
    blist = []
    M = 40
    println(max_coeff)
    for i in 1:M
        c = rand(rng,N)
        Q = rand(rng,N, N)
        # Q = Q * Q' 
        Q = (Q' + Q) / 2
        b1 = abs(randn(rng))
        push!(QList, Q)
        push!(blist, b1)
        push!(clist, c)
        maxcoeff = max(maxcoeff, maximum(abs.(Q)))
        maxcoeff = max(maxcoeff, maximum(abs.(c)))
        maxcoeff = max(maxcoeff, abs.(b1))
    end
    model = Model(Ipopt.Optimizer)

    bits = 12
    @variable(model, x[1:N])
    @objective(model, MIN_SENSE, c_obj' * x + 0.5 * x' * Q_obj * x)
    f0 = to_moi(y, Q_obj, c_obj, 0.0)
    f0q = to_quantized_moi(y, Q_obj, c_obj, 0.0, bits, maxcoeff)
    constlins = []
    constquads = []
    constconst = []

    test = QPBlockData{Float64}()
    testq = QPBlockData{Float64}()
    MOI.set(testq, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), f0q)
    MOI.set(test, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), f0)
    for i in 1:M
        c = clist[i]
        Q = QList[i]
        b1 = blist[i]
        f = to_moi(y, Q, c, b1)
        @constraint(model, 0.5 * x' * Q * x  + c' * x <= b1)
        f = to_moi(y, Q, c, -b1)
        fq = to_quantized_moi(y, Q, c, -b1, bits, maxcoeff)
        MOI.add_constraint(test, f, MOI.LessThan(0.0))
        MOI.add_constraint(testq, fq, MOI.LessThan(0.0))

    end
    optimize!(model)
    # res = solve_pddyn(test, N, tstop=50.,τ=1e-3);
    println("======== Quantized ==============")
    # res3 = solve_pddyn(testq, N, tstop=50.,τ=1e-3, post_sample=false );
    lbounds = vcat([-Inf for i in 1:N],[0 for i in 1:M])
    ubounds = vcat([Inf for i in 1:N],[Inf for i in 1:M])
    res2 = solve_pddyn(test, N, lbounds, ubounds, tstop=30.,τ=1e-3);
    # obj1 = MOI.eval_objective(test, res[3])
    obj2 = MOI.eval_objective(test, res2[3])
    constvec = zeros(M)
    MOI.eval_constraint(test, constvec, res2[3])
    for i in 1:N
        set_start_value(x[i], res2[3][i])
    end
    constraint_indices = []
    for (F, S) in list_of_constraint_types(model)
        # We add a try-catch here because some constraint types might not
        # support getting the primal or dual solution.
        try
            for ci in all_constraints(model, F, S)
                push!(constraint_indices, ci)
                # println(value(ci))
            end
        catch
            @info("Something went wrong getting $F-in-$S. Skipping")
        end
    end
    for i in 1:N
        set_start_value(x[i], res2[3][i])
    end
    for (i, ci) in enumerate(constraint_indices)
        # println(value(ci))
        set_dual_start_value(ci, res2[3][i])
        # set_start_value(ci, constvec[i])
    end
    # for i in 1:M
    #     set_start_value(x[i], res2[3][i])
    # end
    # set_start_values(model, variable_primal_start=res2[3], constraint_dual_start=res2[4])
    println(termination_status(model))
    optimize!(model)
    # obj3 = MOI.eval_objective(test, res3[3])
end

function to_real_rep(mat)
    N = size(mat)[1]
    real_rep = sp.spzeros(2*N, 2*N)
    G = real.(mat)
    B = imag.(mat)
    copyto!(real_rep, CartesianIndices((1:N, 1:N)), G, CartesianIndices((1:N, 1:N)))
    copyto!(real_rep, CartesianIndices((1:N, N+1:2N)), -B, CartesianIndices((1:N, 1:N)))
    copyto!(real_rep, CartesianIndices((N+1:2N, 1:N)), B, CartesianIndices((1:N, 1:N)))
    copyto!(real_rep, CartesianIndices((N+1:2N, N+1:2N)), G, CartesianIndices((1:N, 1:N)))
    return real_rep
end

mutable struct PowerFlowProblem 
    n::Int
    M::Vector{sp.SparseMatrixCSC{Float64, Int}}
    C::Vector{Float64}
    q::Vector{Float64}
    Ψ::Vector{sp.SparseMatrixCSC{Float64, Int}}
    Φ::Vector{sp.SparseMatrixCSC{Float64, Int}}
    p_upper::Vector{Float64}
    p_lower::Vector{Float64}
    q_upper::Vector{Float64}
    q_lower::Vector{Float64}
    p_load::Vector{Float64}
    q_load::Vector{Float64}
    v_mag_min::Vector{Float64}
    v_mag_max::Vector{Float64}
    reference_bus::Int
    function PowerFlowProblem(data::Dict) 
        Y = calc_admittance_matrix(data).matrix
        Ψ = [] # Hermitian components of Y
        Φ = [] # Skew-Hermitian components of Y
        p_upper = []
        p_lower = []
        q_upper = []
        q_lower = []
        p_load = []
        q_load = []
        v_mag_min = []
        v_mag_max = []
        reference_bus = -1
        N = size(Y)[1]
        q = zeros(N)
        C = zeros(N)
        M = []
        # M_inds = []
        # M_
        # C = sp.SparseMatrixCSC{Float64, Int};
        norm = data["baseMVA"]
        for j in 1:N
            eⱼ = zeros(N); eⱼ[j] = 1; Eⱼ = diagm(eⱼ);Ψⱼ = to_real_rep(Eⱼ * Y);Φⱼ = to_real_rep(-im * Eⱼ * Y);
            push!(Ψ, Ψⱼ)
            push!(Φ, Φⱼ)
            push!(p_upper, data["gen"]["$(j)"]["pmax"] / norm)
            push!(q_upper, data["gen"]["$(j)"]["qmax"] / norm)
            push!(p_lower, data["gen"]["$(j)"]["pmin"] / norm)
            push!(q_lower, data["gen"]["$(j)"]["qmin"] / norm)
            push!(p_load, data["load"]["$(j)"]["pd"] / norm)
            push!(q_load, data["load"]["$(j)"]["qd"] / norm)
            push!(v_mag_min, data["bus"]["$(j)"]["vmin"]^2)
            push!(v_mag_max, data["bus"]["$(j)"]["vmax"]^2)
            push!(M, sp.sparse([j, j+N, 2N], [j, j+N, 2N], [1, 1, 0.0]))
            if data["bus"]["$(j)"]["bus_type"] == 3
                reference_bus = j
            end
            if size(data["gen"]["$(j)"]["cost"])[1] > 0
                q[j] = data["gen"]["$(j)"]["cost"][2]
                C[j] = data["gen"]["$(j)"]["cost"][1]
            end
        end
        return new(
            N,
            M,
            C,
            q,
            Ψ,
            Φ,
            p_upper,
            p_lower,
            q_upper,
            q_lower,
            p_load,
            q_load,
            v_mag_min,
            v_mag_max,
            reference_bus
        )
    end
end
function split_variables(p::PowerFlowProblem) 
    V = x[1:2p.N]
    λ̄  = x[2p.N+1:2p.N+N+1]
    λ̄  = x[2p.N+1:2p.N+N+1]
end

function calc_power_gradient(p::PowerFlowProblem,
                             x::Vector{Float64},
                             dx::Vector{Float64})
    n = p.n
    V = x[1:2n]
    λᵘ = x[2n+1:3n]
    λˡ = x[3n+1:4n]
    γᵘ = x[4n+1:5n]
    γˡ = x[5n+1:6n]
    μᵘ = x[6n+1:7n]
    μˡ = x[7n+1:end]
    dV = dx[1:2n]
    dλᵘ = dx[2n+1:3n]
    dλˡ = dx[3n+1:4n]
    dγᵘ = dx[4n+1:5n]
    dγˡ = dx[5n+1:6n]
    dμᵘ = dx[6n+1:7n]
    dμˡ = dx[7n+1:end]
    for (j, (quad_cost, lin_cost, Ψⱼ, Φⱼ, Mⱼ, pd, qd, qmax, qmin, pmax, pmin, vmax, vmin)) in enumerate(zip(
            p.C, p.q, p.Ψ, p.Φ, p.M, p.p_load, p.q_load, p.q_upper, p.q_lower, p.p_upper, p.p_lower, p.v_mag_max, p.v_mag_min))
        p_j = V' * Ψⱼ * V
        q_j = V' * Φⱼ * V
        mag_j = V' * Mⱼ * V
        dV .-= 2 * ( 2 * quad_cost * (p_j + pd) .* Ψⱼ + lin_cost * Ψⱼ + 
                    (λᵘ[j]-λˡ[j]) * Ψⱼ + 
                    (γᵘ[j]-γˡ[j]) * Φⱼ + 
                    (μᵘ[j]-μˡ[j]) * Mⱼ) * V
        dλˡ[j] = pmin - pd - p_j
        dλᵘ[j] = -(pmax - pd - p_j)
        dγˡ[j] = qmin - qd - q_j
        dγᵘ[j] = -(qmax - qd - q_j)
        dμˡ[j] = (vmin - mag_j)
        dμᵘ[j] = -(vmax - mag_j)
    end
    dV[end] -= 2 * V[end]
    dV = min.(dV, 10000.)
    dV = max.(dV, -10000.)
    dx .= vcat(dV, dλˡ, dλᵘ, dγˡ, dγᵘ, dμˡ, dμᵘ)
end
function eval_objective(p::PowerFlowProblem, V::Vector{Float64})
    result = 0.0
    for j in 1:p.n
        quad_cost = p.C[j]
        lin_cost = p.q[j]
        println(V' * p.Ψ[j] * V)
        power_value = V' * p.Ψ[j] * V + p.p_load[j]
        result += quad_cost * power_value^2 + lin_cost * power_value
        
    end
    return result
end
function eval_constraints(p::PowerFlowProblem, V::Vector{Float64})
    dλᵘ = zeros(p.n)
    dλˡ = zeros(p.n)
    dγᵘ = zeros(p.n)
    dγˡ = zeros(p.n)
    dμᵘ = zeros(p.n)
    dμˡ = zeros(p.n)
    for (j, (Ψⱼ, Φⱼ, Mⱼ, pd, qd, qmax, qmin, pmax, pmin, vmax, vmin)) in enumerate(zip(
        p.Ψ, p.Φ, p.M, p.p_load, p.q_load, p.q_upper, p.q_lower, p.p_upper, p.p_lower, p.v_mag_max, p.v_mag_min))
        p_j = V' * Ψⱼ * V
        q_j = V' * Φⱼ * V
        mag_j = V' * Mⱼ * V
        dλˡ[j] = pmin - pd - p_j
        dλᵘ[j] = -(pmax - pd - p_j)
        dγˡ[j] = qmin - qd - q_j
        dγᵘ[j] = -(qmax - qd - q_j)
        dμˡ[j] = (vmin - mag_j)
        dμᵘ[j] = -(vmax - mag_j)
    end
    return vcat(dλᵘ, dλˡ, dγᵘ, dγˡ, dμᵘ, dμˡ)
end

function solve_pddyn(
    problem::PowerFlowProblem;
    tstop::Float64 = 5000.0,
    τ::Float64 = 1e-5,
    verbose::Bool = true,
    log_freq::Int = 10_000,
)
    printf(x::Float64) = Printf.@sprintf("% 1.6e", x)
    printf(x::Int) = Printf.@sprintf("%6d", x)
    n = problem.n
    function grad(dx, x, p, t)
        fill!(dx, 0)
        calc_power_gradient(problem, x, dx)
        x[2n+1:end] .= max.(x[2n+1:end], 0)
        x[1:2n] .= min.(max.(x[1:2n], 0.9), 1.1)
    end
    integrator = RK45Integrator(8n, τ)
    time = 0.0
    prob = ODEProblem(grad, rand(8n), (0, tstop))

    x = rand(8n)
    k = 0
    while time < tstop
        # Take a single gradient step
        # fill!(grad, 0.)
        rks_step!(integrator, x, grad, time)
        x[2n+1:end] .= max.(x[2n+1:end], 0)
        x[1:2n] .= min.(max.(x[1:2n], 0.9), 1.1)
        if verbose && (mod(k, log_freq) == 0)
            testvec = eval_constraints(problem, x[1:2n])
            comp = testvec .* x[2n+1:end]
            logs = printf.((time, eval_objective(problem, x[1:2n]), max(testvec...), norm(comp)))
            println(join(logs, "\t"))
            
        end
        time += τ   
        k += 1
    end
    
    V = x[1:n]
    λᵘ = x[n+1:2n]
    λˡ = x[2n+1:3n]
    γᵘ = x[3n+1:4n]
    γˡ = x[4n+1:5n]
    μᵘ = x[5n+1:6n]
    μˡ = x[6n+1:end]
    return V, λᵘ, λˡ, γᵘ, γˡ, μᵘ, μˡ
end