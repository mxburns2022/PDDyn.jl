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
using Plots
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
        dual,
        lbounds)
    # First get the primal gradient
    MOI.eval_objective_gradient(problem, dprimal, primal)
    _ = MOI.eval_constraint_jacobian(problem, J, primal)
    for ((i, j), val) in zip(structure, J)
        dprimal[j] += ForwardDiff.value(dual[i] * val)
    end
    # Now get the objective gradient
    MOI.eval_constraint(problem, ddual, primal);
    # ddual .= max.(ddual, lbounds[size(dprimal)[1]+1,end])
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
    tstop::Float64 = 10.0,
    τ::Float64 = 1e-5,
    tol::Float64 = 1e-6,
    verbose::Bool = true,
    log_freq::Int = 1000,
    post_sample::Bool = false,
    delay::Float64 = 0.0,
    target::Union{QPBlockData{Float64}, Type{Nothing}} = Nothing
)
    printf(x::Float64) = Printf.@sprintf("% 1.6e", x)
    printf(x::Int) = Printf.@sprintf("%6d", x)
    # target_problem = 
    if target == Nothing
        target = problem
    end
    # Implement primal-dual dynamics using the differential equation solver
    m = size(problem.constraints)[1]
    # n = problem.parameters
    Jstructure = MOI.jacobian_structure(problem)
    J = zeros(size(Jstructure));
    # copyto!(primal_dual_system, dual_dual_inds, -P, CartesianIndices(P))
    # copyto!(primal_dual_system, primal_dual_inds, A', CartesianIndices(A'))
    # Projected gradient descent on Lagrangian
    #       L(x, \lambda) = 
    print(lbounds)
    inequality_indices = Int[]
    for i in n+1:n+m
        if lbounds[i] == 0.0
            push!(inequality_indices, i)
        end
    end
    println(inequality_indices)
    function condition(u, t, integrator)
        true
    end
    
    function affect!(integrator)

        integrator.u .= min.(max.(integrator.u, lbounds), ubounds)
    end
    callback = DiscreteCallback(condition, affect!)

    function grad(dx, x,  t)
        dprimal = zeros(n)
        ddual = zeros(m)
        get_grad(problem, J, Jstructure, dprimal, ddual, x[1:n], x[n+1:end],lbounds)
        dx[1:n] .= -dprimal
        dx[n+1:end] .= ddual
        # ineq_grad = dx[inequality_indices] 
        # ineq_vars = x[inequality_indices] 
        x.= min.(max.(x, lbounds), ubounds)
        # dx[inequality_indices] .*= convert.(Float64, max.((ineq_grad .> 0.), ineq_vars .> 0.))
        # x[n+1:end] .= max.(0., x[n+1:end])

    end
    time = 0.0
    x = randn(n+m)
    fill(x[m+1:end], 0)
    x_copy = copy(x)
    next_update_time = delay
    integrator = RK45Integrator(n+m, τ)
    time = 0.0
    # x = rand(n+m)
    k = 0
    status = MOI.OTHER_ERROR
    prev_fval = Inf64
    dx = zeros(size(x)...)
    while status == MOI.OTHER_ERROR && time < tstop
    # for i in 1:100
        if delay < τ
            # Take a single gradient step
            rks_step!(integrator, x, grad, time)
        else
            fill!(dx, 0.0)
            grad(dx, x_copy, time)
            x += τ * (dx)
            if time > next_update_time
                x_copy = copy(x)
                next_update_time += delay
            end
        end
        # odeproblem = ODEProblem(grad, x, (0.0, tstop/100))
        # result = solve(odeproblem, DP8(), save_everystep=false, callback=callback);
        x.= min.(max.(x, lbounds), ubounds)
        # x = result.u[end]
        if time >= tstop
            status = MOI.ITERATION_LIMIT
        end
        if  (mod(k, log_freq) == 0 || status != MOI.OTHER_ERROR)
            testvec = fill(0.0, m);
            MOI.eval_constraint(problem, testvec, x[1:n])
            comp = testvec .* x[n+1:end]
            objective = MOI.eval_objective(problem, x[1:n])
            if verbose
                logs = printf.((time, MOI.eval_objective(problem, x[1:n]), maximum(testvec), norm(comp)))
                # println(testvec .* x[n+1:end])
                println(join(logs, "\t"))
            end
            if (norm(comp) < tol) && abs(objective-prev_fval) < tol &&  maximum(testvec) < tol
                break
            end
            prev_fval = objective
        end
        time += τ   
        k += 1
        # break
        # break
    end
    testvec = fill(0.0, m);
    MOI.eval_constraint(target, testvec, x[1:n])
    objective = MOI.eval_objective(target, x[1:n])
    status = MOI.LOCALLY_SOLVED
    constvec = zeros(m)
    MOI.eval_constraint(target, constvec, x[1:n])
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
    else
        dual_status = MOI.INFEASIBLE_POINT
    end
        
    return status, primal_status, dual_status, time, objective, x[1:n], x[n+1:m+n]
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
    nbr::Int
    M::Vector{sp.SparseMatrixCSC{Float64, Int}}
    C::Vector{Float64}
    q::Vector{Float64}
    Ψ::Vector{sp.SparseMatrixCSC{Float64, Int}}
    Ψbr::Vector{sp.SparseMatrixCSC{Float64, Int}}
    Φ::Vector{sp.SparseMatrixCSC{Float64, Int}}
    Φbr::Vector{sp.SparseMatrixCSC{Float64, Int}}
    p_upper::Vector{Float64}
    p_lower::Vector{Float64}
    q_upper::Vector{Float64}
    q_lower::Vector{Float64}
    p_load::Vector{Float64}
    q_load::Vector{Float64}
    v_mag_min::Vector{Float64}
    v_mag_max::Vector{Float64}
    s_max::Vector{Float64}
    reference_bus::Int
    max_coeff::Float64
    function PowerFlowProblem(data::Dict) 
        Y = calc_admittance_matrix(data).matrix
        Ψ = [] # Hermitian components of Y
        Ψbr = [] # Hermitian components of Y (branchess)
        Φ = [] # Skew-Hermitian components of Y
        Φbr = [] # Skew-Hermitian components of Y
        p_upper = []
        p_lower = []
        q_upper = []
        q_lower = []
        p_load = []
        q_load = []
        v_mag_min = []
        v_mag_max = []
        s_max = []
        reference_bus = -1
        N = size(Y)[1]
        q = zeros(N)
        C = zeros(N)
        M = []
        Nbr = length(data["branch"])
        # M_inds = []
        # M_
        # C = sp.SparseMatrixCSC{Float64, Int};
        norm = data["baseMVA"]
        bus_ids = Dict{Int, Int}()
        bus_ids_inv = Dict{Int, Int}()
        keyvalues = sort([parse(Int, i) for i in keys(data["bus"])])
        for j in 1:N
            bus_ids[keyvalues[j]] = j
            bus_ids_inv[j] = keyvalues[j]
        
        end
        print(bus_ids_inv)
        for j in 1:N
            eⱼ = zeros(N); eⱼ[j] = 1; Eⱼ = diagm(eⱼ);Ψⱼ = to_real_rep( 0.5 * ((Eⱼ*Y)'+(Eⱼ*Y)));Φⱼ = to_real_rep(-im * 0.5 * ((Eⱼ*Y)'-(Eⱼ*Y)));
            push!(Ψ, Ψⱼ)
            push!(Φ, Φⱼ)
            bus_key = "$(bus_ids_inv[j])"
            if bus_key in keys(data["gen"])
                push!(p_upper, data["gen"][bus_key]["pmax"])
                push!(q_upper, data["gen"][bus_key]["qmax"])
                push!(p_lower, data["gen"][bus_key]["pmin"])
                push!(q_lower, data["gen"][bus_key]["qmin"])
                if size(data["gen"][bus_key]["cost"])[1] > 0
                    q[j] = data["gen"][bus_key]["cost"][2]
                    C[j] = data["gen"][bus_key]["cost"][1]
                end
            else 
                push!(p_upper, 1e10)
                push!(q_upper, 1e10)
                push!(p_lower, -1e10)
                push!(q_lower, -1e10)
            end
            if haskey(data["bus"], bus_key)
                push!(v_mag_max, data["bus"][bus_key]["vmax"]^2)
                push!(v_mag_min, data["bus"][bus_key]["vmin"]^2)
                if data["bus"][bus_key]["bus_i"] == 3
                    reference_bus = j
                end
            else
                push!(v_mag_min, 0)
                push!(v_mag_max, 1e10)
            end
            if bus_key in keys(data["load"])
                push!(p_load, data["load"][bus_key]["pd"])
                push!(q_load, data["load"][bus_key]["qd"])
            else
                push!(p_load, 0)
                push!(q_load, 0)
            end
            
            push!(M, sp.sparse([j, j+N, 2N], [j, j+N, 2N], [1, 1, 0.0]))
            
        end
        Yh = 0.5 * (Y + Y');
        Y_sk = -im * 0.5 * (Y' - Y);
        for (_, brdata) in pairs(data["branch"])
            srcnode = bus_ids[brdata["f_bus"]]
            destnode = bus_ids[brdata["t_bus"]]
            eᵢ = zeros(N); eᵢ[srcnode] = 1; Eᵢ = diagm(eᵢ);
            eⱼ = zeros(N); eⱼ[destnode] = 1; Eⱼ = diagm(eⱼ);

            Yij_h = sp.spzeros(ComplexF64, size(Y))
            Yij_h[srcnode, srcnode] =  -(Yh[srcnode, destnode])
            Yij_h[srcnode, destnode] = -(Yh[srcnode, destnode])
            # Yij_h[destnode, srcnode] = (Yh[srcnode,  destnode])

            Yij_sk = sp.spzeros(ComplexF64, size(Y))
            Yij_sk[srcnode, srcnode] =  -(Y_sk[srcnode, destnode])
            Yij_sk[srcnode, destnode] = (Y_sk[srcnode, destnode])
            # Yij[srcnode, destnode] = -Y[srcnode, destnode]
            # Φⱼ = sp.spzeros(size(Y))
            # Yij = (Eᵢ + Eⱼ) * Y * (Eᵢ + Eⱼ)
            # Yij[srcnode, srcnode] = Y[srcnode, srcnode]
            # Yij[destnode, destnode] = Y[destnode, destnode]
            Ψⱼ = to_real_rep(Yij_h);
            Φⱼ = to_real_rep(Yij_sk);
            # Ψⱼ[srcnode, srcnode] = ΨT[srcnode, srcnode]
            # Ψⱼ[srcnode+N, srcnode+N] = ΨT[srcnode+N, srcnode+N]
            # Ψⱼ[srcnode, destnode] = ΨT[srcnode, destnode]
            # Ψⱼ[destnode, srcnode] = ΨT[destnode, srcnode]
            # Ψⱼ[srcnode+N, destnode+N] = ΨT[srcnode+N, destnode+N]
            # Ψⱼ[destnode+N, srcnode+N] = ΨT[destnode+N, srcnode+N]
            # Ψⱼ[srcnode, destnode+N] = ΨT[srcnode, destnode+N]
            # Ψⱼ[destnode+N, srcnode] = ΨT[destnode+N, srcnode]
            # Ψⱼ[srcnode+N, destnode] = ΨT[srcnode+N, destnode]
            # Ψⱼ[destnode, srcnode+N] = ΨT[destnode, srcnode+N]

            # Φⱼ[srcnode, srcnode] = ΦT[srcnode, srcnode]
            # Φⱼ[srcnode+N, srcnode+N] = ΦT[srcnode+N, srcnode+N]
            # Φⱼ[srcnode, destnode] = ΦT[srcnode, destnode]
            # Φⱼ[destnode, srcnode] = ΦT[destnode, srcnode]
            # Φⱼ[srcnode+N, destnode+N] = ΦT[srcnode+N, destnode+N]
            # Φⱼ[destnode+N, srcnode+N] = ΦT[destnode+N, srcnode+N]
            # Φⱼ[srcnode, destnode+N] = ΦT[srcnode, destnode+N]
            # Φⱼ[destnode+N, srcnode] = ΦT[destnode+N, srcnode]
            # Φⱼ[srcnode+N, destnode] = ΦT[srcnode+N, destnode]
            # Φⱼ[destnode, srcnode+N] = ΦT[destnode, srcnode+N]


            # Ψⱼ[destnode, destnode] = 0
            # Ψⱼ[destnode+N, destnode+N] = 0
            # Ψⱼ[srcnode, srcnode] = 0
            push!(Ψbr, Ψⱼ)
            push!(Φbr, Φⱼ)
            push!(s_max, brdata["rate_a"]^2)
        end
        function maxval(y)
            if abs(y) > 1e8
                return 0
            end
            return abs(y)
        end
        max_coeff = maximum([maximum([maximum(maxval.(j)) for j in i]) for i in [Ψ, Φ, M]])
        max_coeff = max(maximum(maxval.(vcat(p_upper, p_lower, q_lower, q_upper, v_mag_max, v_mag_min))))
        return new(
            N,
            Nbr,
            M,
            C,
            q,
            Ψ,
            Ψbr,
            Φ,
            Φbr,
            p_upper,
            p_lower,
            q_upper,
            q_lower,
            p_load,
            q_load,
            v_mag_min,
            v_mag_max,
            s_max,
            reference_bus,
            max_coeff
        )
    end
    function PowerFlowProblem(
        N,
        Nbr,
        M,
        C,
        q,
        Ψ,
        Ψbr,
        Φ,
        Φbr,
        p_upper,
        p_lower,
        q_upper,
        q_lower,
        p_load,
        q_load,
        v_mag_min,
        v_mag_max,
        s_max,
        reference_bus,
        max_coeff)
        return new(
            N,
            Nbr,
            M,
            C,
            q,
            Ψ,
            Ψbr,
            Φ,
            Φbr,
            p_upper,
            p_lower,
            q_upper,
            q_lower,
            p_load,
            q_load,
            v_mag_min,
            v_mag_max,
            s_max,
            reference_bus,
            max_coeff
        )
end
end

function quantize(p::PowerFlowProblem, bits::Int)

        scale = if (bits > 0) 1 / (2^bits) else 0.0 end
        Y = calc_admittance_matrix(data).matrix
        Ψ = [] # Hermitian components of Y
        Ψbr = [] # Hermitian components of Y (branchess)
        Φ = [] # Skew-Hermitian components of Y
        Φbr = [] # Skew-Hermitian components of Y
        p_upper = []
        p_lower = []
        q_upper = []
        q_lower = []
        p_load = []
        q_load = []
        v_mag_min = []
        v_mag_max = []
        s_max = quantize.(p.s_max, scale, p.max_coeff, false)
        reference_bus = p.reference_bus
        N = size(Y)[1]
        q = quantize.(p.q, scale, p.max_coeff, false)
        C = quantize.(p.C, scale, p.max_coeff, false)
        M = []
        Nbr = length(p.Φbr)
        for (j, (Ψⱼ, Φⱼ, Mⱼ, pd, qd, qmax, qmin, pmax, pmin, vmax, vmin)) in enumerate(zip(
            p.Ψ, p.Φ, p.M, p.p_load, p.q_load, p.q_upper, p.q_lower, p.p_upper, p.p_lower, p.v_mag_max, p.v_mag_min))
            push!(Ψ, quantize.(Ψⱼ, scale, p.max_coeff, false))
            push!(Φ, quantize.(Φⱼ, scale, p.max_coeff, false))
            
            push!(p_upper, if pmax < 1e8 quantize(pmax, scale, p.max_coeff, false) else 1e10 end)
            push!(p_lower, if pmin > -1e8 quantize(pmin, scale, p.max_coeff, false) else -1e10 end)
            push!(q_upper, if qmax < 1e8 quantize(qmax, scale, p.max_coeff, false) else 1e10 end)
            push!(q_lower, if qmin > -1e8 quantize(qmin, scale, p.max_coeff, false) else -1e10 end)
            push!(p_load, quantize(pd, scale, p.max_coeff, false))
            push!(q_load, quantize(qd, scale, p.max_coeff, false))
            push!(v_mag_min, quantize(vmin, scale, p.max_coeff, false))
            push!(v_mag_max, quantize(vmax, scale,p.max_coeff, false))
            push!(M, quantize.(Mⱼ, scale,p.max_coeff, false))
        end
        for (j, (Ψⱼ, Φⱼ, sⱼ)) in enumerate(zip(p.Ψbr, p.Φbr, p.s_max)) 
            push!(Ψbr, quantize.(Ψⱼ, scale, p.max_coeff, false))
            push!(Φbr, quantize.(Φⱼ, scale, p.max_coeff, false))
        end
        return PowerFlowProblem(
            N,
            Nbr,
            M,
            C,
            q,
            Ψ,
            Ψbr,
            Φ,
            Φbr,
            p_upper,
            p_lower,
            q_upper,
            q_lower,
            p_load,
            q_load,
            v_mag_min,
            v_mag_max,
            s_max,
            reference_bus, 
            1.0
        )
end

function get_model(p::PowerFlowProblem)
    model = Model(Ipopt.Optimizer)
    @variable(model, V[1:2p.n])
    for (j, (Ψⱼ, Φⱼ, Mⱼ, pd, qd, qmax, qmin, pmax, pmin, vmax, vmin)) in enumerate(zip(
        p.Ψ, p.Φ, p.M, p.p_load, p.q_load, p.q_upper, p.q_lower, p.p_upper, p.p_lower, p.v_mag_max, p.v_mag_min))

        p_j = V' * Ψⱼ * V
        q_j = V' * Φⱼ * V
        mag_j = V' * Mⱼ * V
        @constraint(model, pmin <= (pd + p_j) <= pmax )
        @constraint(model, qmin <= (qd + q_j) <= qmax )
        # @constraint(model, vmin <= mag_j <= vmax )
    end
    # @constraint(model, V[p.n+p.reference_bus] == 0 )
    @objective(model, MIN_SENSE, 0)
    return model
end

function calc_power_gradient(p::PowerFlowProblem,
                             x::AbstractVector,
                             dx::AbstractVector)
    n = p.n
    V = x[1:2n]
    λᵘ = x[2n+1:3n]
    λˡ = x[3n+1:4n]
    γᵘ = x[4n+1:5n]
    γˡ = x[5n+1:6n]
    μᵘ = x[6n+1:7n]
    μˡ = x[7n+1:8n]
    ν = x[8n+1:end]

    dV = dx[1:2n]
    dλᵘ = dx[2n+1:3n]
    dλˡ = dx[3n+1:4n]
    dγᵘ = dx[4n+1:5n]
    dγˡ = dx[5n+1:6n]
    dμᵘ = dx[6n+1:7n]
    dμˡ = dx[7n+1:8n]
    dν = dx[8n+1:end]
    # dνˡᵢ = dx[10n+1:11n]
    # dνᵘᵢ = dx[11n+1:12n]
    for (j, (quad_cost, lin_cost, Ψⱼ, Φⱼ, Mⱼ, pd, qd, qmax, qmin, pmax, pmin, vmax, vmin)) in enumerate(zip(
            p.C, p.q, p.Ψ, p.Φ, p.M, p.p_load, p.q_load, p.q_upper, p.q_lower, p.p_upper, p.p_lower, p.v_mag_max, p.v_mag_min))
        p_j = V' * Ψⱼ * V
        q_j = V' * Φⱼ * V
        mag_j = V[j]^2 + V[j+n]^2
        # $$ \frac{}{} $$
        dV .-=  2(2 * quad_cost * (p_j + pd) .*(Ψⱼ) + lin_cost .* (Ψⱼ) + 
                    (λᵘ[j]-λˡ[j]) * (Ψⱼ) + 
                    (γᵘ[j]-γˡ[j]) * (Φⱼ) + 
                    (μᵘ[j]-μˡ[j]) * Mⱼ) * V
        # dV .-= ((λᵘ[j]-λˡ[j]) * (Ψⱼ + Ψⱼ') + 
                #    (γᵘ[j]-γˡ[j]) * (Φⱼ + Φⱼ') + 
                    # (μᵘ[j]-μˡ[j]) * (Mⱼ + Mⱼ')) * V
        # dV[j] -= 2(#(λᵘ[j]-λˡ[j]) * (Ψⱼ + Ψⱼ') + 
                    #   (γᵘ[j]-γˡ[j]) * (Φⱼ + Φⱼ') + 
                    # (μᵘ[j]-μˡ[j])) * V[j]
        # dV[j+n] -= 2(#(λᵘ[j]-λˡ[j]) * (Ψⱼ + Ψⱼ') + 
                    #   (γᵘ[j]-γˡ[j]) * (Φⱼ + Φⱼ') + 
                    # (μᵘ[j]-μˡ[j])) * V[j+n]
        dλˡ[j] = pmin - pd - p_j
        dλᵘ[j] = pd + p_j - pmax
        dγˡ[j] = qmin - qd - q_j
        dγᵘ[j] = qd + q_j - qmax
        dμˡ[j] = vmin - mag_j
        dμᵘ[j] = mag_j - vmax
    end
    # for (j, (Ψⱼ, Φⱼ, sⱼ)) in enumerate(zip(p.Ψbr, p.Φbr, p.s_max)) 
    #     p_br_j = V' * Ψⱼ * V
    #     q_br_j = V' * Φⱼ * V
    #     # dV .-=  4ν[j].*(p_br_j*(Ψⱼ) + q_br_j * (Φⱼ)) * V
    #     # dν[j] = p_br_j^2 + q_br_j^2 - sⱼ
    # end
    dλˡ .*= convert.(Float64, max.((dλˡ .> 0.), λˡ .> 0.))
    dγˡ .*= convert.(Float64, max.((dγˡ .> 0.), γˡ .> 0.))
    dμˡ .*= convert.(Float64, max.((dμˡ .> 0.), μˡ .> 0.))
    dγᵘ .*= convert.(Float64, max.((dγᵘ .> 0.), γᵘ .> 0.))
    dλᵘ .*= convert.(Float64, max.((dλᵘ .> 0.), λᵘ .> 0.))
    dμᵘ .*= convert.(Float64, max.((dμᵘ .> 0.), μᵘ .> 0.))
    dν .*= convert.(Float64, max.((dν .> 0.), ν .> 0.))
    dV[n+1] -= 2 * V[n+1]
    dx .= vcat(dV, dλᵘ, dλˡ,  dγᵘ,dγˡ, dμᵘ,  dμˡ, dν)
end
function eval_objective(p::PowerFlowProblem, V::Vector{Float64})
    result = 0.0
    for j in 1:p.n
        quad_cost = p.C[j]
        lin_cost = p.q[j]
        power_value = V' * p.Ψ[j] * V + p.p_load[j]
        result += quad_cost * power_value^2 + lin_cost * power_value
        println("$(quad_cost), $(lin_cost), $(power_value)")
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
    dν = zeros(p.nbr)
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
    for (j, (Ψⱼ, Φⱼ, sⱼ)) in enumerate(zip(p.Ψbr, p.Φbr, p.s_max)) 
        p_br_j = V' * Ψⱼ * V
        q_br_j = V' * Φⱼ * V
        dν[j] = p_br_j^2 + q_br_j^2 - sⱼ
    end
    return vcat(dλᵘ, dλˡ, dγᵘ, dγˡ, dμᵘ, dμˡ, dν)
end

function solve_pddyn(
    problem::PowerFlowProblem;
    tstop::Float64 = 30000.0,
    τ::Float64 = 1e-5,
    verbose::Bool = true,
    log_freq::Int = 10_000,
    bits::Int = 0,
)
    printf(x::Float64) = Printf.@sprintf("% 1.6e", x)
    printf(x::Int) = Printf.@sprintf("%6d", x)
    # problem = quantize(p, bits)
    n = problem.n
    function grad(dx, x, p, t)
        fill!(dx, 0)
        # x[1:2n] .= min.(max.(x[1:2n], -1.1), 1.1)
        x[2n+1:end] .= max.(x[2n+1:end], 0)
        calc_power_gradient(problem, x, dx)
    end
    x = rand(8n + problem.nbr)
    # x[1:2n] = [1.1000000050616032, 1.0864554011184095, 1.0773745270552761, 4.6790506294074145e-35, 0.17208914221302823, -0.22195526495444665]
    x[2n+1:end] .= 0
    result = Nothing
    for i in 1:1
        τ_stop = tstop / 1
        odeproblem = ODEProblem(grad, x, (0.0, τ_stop))
        result = solve(odeproblem, TRBDF2(), save_everystep=true);
        fvals = []
        fvals_fake = []
        tvals = []
        for (t, x) in zip(result.t, result.u)
            # print(testvec)
            eval = maximum(eval_constraints(p, x[1:2n]))#eval_objective(problem, x[1:2n])
            push!(fvals, max(maximum(eval_constraints(p, x[1:2n])), 1e-10))
            push!(fvals_fake, max(maximum(eval_constraints(problem, x[1:2n])), 1e-10))
            push!(tvals, t)
            # logs = printf.((i * τ_stop, eval_objective(problem, x[1:2n]), max(testvec...), norm(comp)))
            # println(testvec .* x[n+1:end])
            # println(join(logs, "\t"))
            # plot!()
        end
        pltval = plot([tvals, tvals], [fvals, fvals_fake], yscale=:log10,ylabel="Infeasibility", xlabel="Time")
        display(pltval)
        x = result.u[end]
        testvec = eval_constraints(problem, x[1:2n])
        println(x)
        println("Constraints: ", maximum(testvec))
        println("Objective: ", eval_objective(problem, x[1:2n]))
        
    # println(fvals[end])
    end

    # x = rand(8n)
    # k = 0
    
    # while time < tstop
    #     # Take a single gradient step
    #     # fill!(grad, 0.)
    #     rks_step!(integrator, x, grad, time)
    #     x[2n+1:end] .= max.(x[2n+1:end], 0)
    #     x[1:2n] .= min.(max.(x[1:2n], 0.9), 1.1)
    #     if verbose && (mod(k, log_freq) == 0)
    #         testvec = eval_constraints(problem, x[1:2n])
    #         comp = testvec .* x[2n+1:end]
    #         logs = printf.((time, eval_objective(problem, x[1:2n]), max(testvec...), norm(comp)))
    #         println(join(logs, "\t"))
            
    #     end
    #     time += τ   
    #     k += 1
    # end
    
    V = x[1:2n]
    λᵘ = x[2n+1:4n]
    λˡ = x[3n+1:4n]
    γᵘ = x[4n+1:5n]
    γˡ = x[5n+1:6n]
    μᵘ = x[6n+1:7n]
    μˡ = x[7n+1:8n]
    ν = x[8n+1:end]
    return V, λᵘ, λˡ, γᵘ, γˡ, μᵘ, μˡ,ν,result
end
