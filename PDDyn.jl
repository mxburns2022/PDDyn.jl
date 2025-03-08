import MathOptInterface as MOI
import SparseArrays as Sp
import Printf
using JuMP
using Ipopt
using LinearAlgebra
using Random
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
    n::Int;
    tstop::Float64 = 50.0,
    τ::Float64 = 1e-5,
    tol::Float64 = 1e-5,
    verbose::Bool = true,
    log_freq::Int = 1_000,
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
            if m > 0
                testvec = fill(0.0, m);
                MOI.eval_constraint(problem, testvec, x[1:n])
                comp = testvec .* x[n+1:end]
                print(testvec)
                logs = printf.((time, MOI.eval_objective(problem, x[1:n]), max(testvec...), norm(comp)))
                # println(testvec .* x[n+1:end])
                println(join(logs, "\t"))
            else
                logs = printf.((time, MOI.eval_objective(problem, x[1:n])))
                # println(testvec .* x[n+1:end])
                println(join(logs, "\t"))
            end
        end
        time += τ   
        k += 1
    end
    if post_sample
        grad_buff = zeros(n+m)
        scratch_vals = zeros(m)
        n_iter = 50000
        best_obj = MOI.eval_objective(target, x[1:n])
        bestx = zeros(n+m)
        copy!(bestx, x)
        for i in 1:n_iter
            grad(grad_buff, x, time)
            grad_buff[1:n] .+= 1 / 256 *( sqrt( 1 / 1e-4)) * randn(n)
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
                # comp = testvec .* scratch_vals[n+1:end]./i
                # logs = printf.((time, MOI.eval_objective(problem, scratch_vals[1:n]./i), max(testvec...), norm(comp)))
                # println(testvec .* x[n+1:end])
                # println(join(logs, "\t"))
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


function solve_pddyn_block(
    problem::QPBlockData{Float64},
    n::Int,
    blocks::Int,
    tol::Float64 = 1e-8,
    verbose::Bool = true,
    log_freq::Int = 1_000,
    τ::Float64 = 1e-1,
    epoch::Float64 = 1e-3,
    tstop::Float64 = 50.0
)
    bsize = convert(Int, floor(n/blocks));
    rem = n % blocks;
    sizes = vcat([bsize + 1 for i in 1:rem], [bsize  for i in rem+1:blocks])
    sizes = cumsum(sizes)
    n_partitions = vcat([1:sizes[1]], [sizes[i]:sizes[i+1] for i in 1:length(sizes)-1])
    printf(x::Float64) = Printf.@sprintf("% 1.6e", x)
    printf(x::Int) = Printf.@sprintf("%6d", x)
    # Implement primal-dual dynamics using the differential equation solver
    m = size(problem.constraints)[1]
    bsize = convert(Int, floor(m/blocks));
    rem = m % blocks;
    sizes = vcat([bsize + 1 for i in 1:rem], [bsize  for i in rem+1:blocks])
    if m < blocks 
        sizes = [bsize + 1 for i in 1:rem]
    end
    sizes = cumsum(sizes)
    m_partitions = vcat([n+1:n+sizes[1]+1], [n+sizes[i]:n+sizes[i+1] for i in 1:length(sizes)-1])
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
    currblock_n = 1
    currblock_m = 1
    println(n, " ", m, " ",sizes, " ",n+m)
    println(n_partitions)
    println(m_partitions, m, n+m)
    # return
    while status == MOI.OTHER_ERROR && time < tstop
        τblock = 0.0

        while τblock < epoch
            rks_step!(integrator, x, grad, time + τblock, [n_partitions[currblock_n], m_partitions[currblock_m]])
            x[n+1:end] .= max.(0., x[n+1:end])
            τblock += τ
        end
        currblock_n = abs(rand(rng,Int)) % length(n_partitions) + 1
        currblock_m = abs(rand(rng,Int)) % length(m_partitions) + 1
        # Take a single gradient step
        # rks_step!(integrator, x, grad, time)
        # x[n+1:end] .= max.(0., x[n+1:end])
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
        time +=  epoch
        k += 1
    end
    return status, time, x[1:n], x[n+1:m+n], MOI.eval_objective(problem, x[1:n])
end
function main_test()
    N = 50
    y = MOI.VariableIndex.(1:N)

    c_obj = rand(rng,N)
    # Q_obj = Sp.spdiagm(N, N, ones(N))
    Q_obj = diagm(N, N, ones(N))
    maxcoeff = max(maximum(abs.(c_obj)), maximum(abs.(Q_obj)))
    QList = []
    clist = []
    blist = []
    M = 40
    for i in 1:M
        global maxcoeff
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

    res2 = solve_pddyn(testq, N, tstop=30.,τ=1e-3);
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