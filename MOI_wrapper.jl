include("PDDyn.jl")
using PowerModels
import LinearAlgebra as linalg
import MathOptInterface.Utilities as MOIU
import SparseArrays as sp
using Random




mutable struct Optimizer <: MOI.AbstractOptimizer
    # Structures storing problem data
    dim::Int
    qpblock::QPBlockData{Float64}
    x_to_index::Dict{MOI.VariableIndex, Int}
    ci_to_index::Dict{MOI.ConstraintIndex, Int}

    # Integrator information
    τ::Float64
    iterations::Int

    # Problem state
    variable_ubounds::Vector{Float64}
    variable_lbounds::Vector{Float64}
    primal::Vector{Float64}
    dual::Vector{Float64}
    solve_time::Float64
    obj_value::Float64
    status::MOI.TerminationStatusCode
    dual_status::MOI.ResultStatusCode
    primal_status::MOI.ResultStatusCode
    rcount::MOI.ResultCount
    silent::Bool;

    function Optimizer() 
        return new(
            0,
            QPBlockData{Float64}(),
            Dict{MOI.VariableIndex, Int}(),
            Dict{MOI.ConstraintIndex, Int}(),
            1e-4,
            1_000,
            Float64[],
            Float64[],
            Float64[],
            Float64[],
            0.0,
            0.0,
            MOI.OPTIMIZE_NOT_CALLED
        )
    end

end

function Base.summary(io::IO, model::Optimizer)
    return print(io, "Primal-Dual Dynamics Solver with $(model.dim) variables and $(size(model.qpblock.constraints)[1]) constraints")
end



function MOI.get(model::Optimizer, ::MOI.Silent)
    return model.silent;
end
function MOI.get(model::Optimizer, ::MOI.SolverName)
    return "Primal-Dual Dynamics";
end

function MOI.set(model::Optimizer, ::MOI.Silent, v::Bool)
    model.silent = v
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64}}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}}},
    ::Type{<:Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64}}},
)
    return true
end


function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
)
    return true
end

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.qpblock)
end

function MOI.is_empty(model::Optimizer)
    # You might want to check every field, not just a few
    return MOI.is_empty(model.qpblock) && model.status == MOI.OPTIMIZE_NOT_CALLED
end
function MOI.empty!(model::Optimizer)
    MOI.empty!(model.qpblock)
    empty!(model.primal)
    empty!(model.dual)
    model.status = MOI.OPTIMIZE_NOT_CALLED
    model.iterations = 0
    model.solve_time = 0.0
    model.obj_value = 0.0
end
function MOI.get(model::Optimizer, ::MOI.DualStatus)
    return model.dual_status
end
function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    return model.primal_status
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.status == MOI.LOCALLY_SOLVED
        return "found a locally optimal primal-dual solution within provided tolerances"
    end
    return "failed to solve"
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if model.status == MOI.LOCALLY_SOLVED && model.primal_status == MOI.FEASIBLE_POINT && model.dual_status == MOI.FEASIBLE_POINT
        return 1
    end
    return 0
end
function MOI.add_constraint(
        model::Optimizer, 
        cons::Union{MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}}, 
        constraintset::Union{MOI.EqualTo{Float64}, MOI.LessThan{Float64}, MOI.GreaterThan{Float64}})
        println(cons, constraintset)
    MOI.add_constraint(model.qpblock, cons, constraintset)
end
function MOI.add_constraint(
    model::Optimizer, 
    cons::MOI.VariableIndex, 
    constraintset::Union{MOI.EqualTo{Float64}, MOI.LessThan{Float64}, MOI.GreaterThan{Float64}})
    v = cons.value
    _, l, u,_,_ = _set_info(constraintset)
    model.variable_ubounds[v] = u
    model.variable_lbounds[v] = l
end
function MOI.set(
        model::Optimizer, 
        attr::MOI.ObjectiveFunction{F}, 
        func::F) where {F}
    MOI.set(model.qpblock, attr, func)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, x::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.primal[model.x_to_index[x]]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros})
    MOI.check_result_index_bounds(model, attr)
    return -model.dual[model.ci_to_index[ci]]
end

MOI.Utilities.@product_of_sets(SetOfZeros, MOI.Zeros) # Set up the set of allowable constraint values (i.e. Ax+b ∈ {0})
const CacheModel = MOI.Utilities.GenericModel{
    Float64, # Coeff type
    MOI.Utilities.ObjectiveContainer{Float64}, # Default Objective Container

    MOI.Utilities.VariablesContainer{Float64}, # Default Variables Container

    # Matrix of constraints to represent `A * x + b in K` for set K
    MOI.Utilities.MatrixOfConstraints{
        # Float64 coefficients
        Float64,
        # Sparse CSC matrix w/ Float64 coeffs with int64 row/col indices and one-based indexing (darn)
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing # Note, if interfacing with C then use zero based indexing with Cdouble and Cint fields
        },
        # Set 'b' to be a Julia vector
        Vector{Float64},
        # Set K of allowable values
        SetOfZeros{Float64}
    },
}



function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike) 
    list_of_indices = MOI.get(src, MOI.ListOfVariableIndices())
    # index_map = MOI.IndexMap() MOI.copy_to(cmodel, src)

    N = MOI.get(src, MOI.NumberOfVariables())
    resize!(dest.variable_lbounds, N)
    fill!(dest.variable_lbounds, -Inf)
    resize!(dest.variable_ubounds, N)
    fill!(dest.variable_ubounds, Inf)
    index_map = MOI.IndexMap()
    for k in list_of_indices
        dest.x_to_index[k] = k.value
        index_map[k] = k
    end
    println("N", N)
    dest.silent = false;
    ftype = MOI.get(src, MOI.ObjectiveFunctionType())
    objfunc = MOI.get(src, MOI.ObjectiveFunction{ftype}())

    maxterm = max_coeff(objfunc)
    qterms = MOI.ScalarQuadraticTerm{Float64}[]
    for t in list_of_indices
        qterm_st = MOI.ScalarQuadraticTerm{Float64}(2, t, t)
        push!(qterms, qterm_st)

    end
    dummylin = MOI.ScalarAffineTerm(0.0, list_of_indices[1])
    objfunc_type = typeof(objfunc)
    # if objfunc_type == MOI.ScalarAffineFunction{Float64}
        
    # end
    # objfunc = objfunc + MOI.ScalarQuadraticFunction(qterms[3:3], [dummylin], 0.0)
    MOI.set(dest.qpblock, MOI.ObjectiveFunction{ftype}(), objfunc)
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        lbound_value = 0.0
        if S <: MOI.EqualTo{Float64}
            lbound_value = -Inf 
        end
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
            if !(F <: MOI.VariableIndex)
                push!(dest.variable_lbounds, lbound_value)
                push!(dest.variable_ubounds, Inf)
            end
            # println()
            func = MOI.get(src, MOI.ConstraintFunction(), ci)
            set = MOI.get(src, MOI.ConstraintSet(), ci)
            MOI.add_constraint(dest, func, set)
        end
    end


    # normalize(dest.qpblock)
    dest.status, dest.primal_status, dest.dual_status, dest.solve_time, dest.primal, dest.dual = solve_pddyn(dest.qpblock,
                                                                                                             N, 
                                                                                                             dest.variable_lbounds, 
                                                                                                             dest.variable_ubounds,
                                                                                                             verbose=!dest.silent,
                                                                                                             τ=1e-8,
                                                                                                             tstop=100000.,
                                                                                                             log_freq=500_000)
    dest.obj_value = MOI.eval_objective(dest.qpblock, dest.primal)
    return index_map, false
end

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.status
MOI.get(model::Optimizer, ::MOI.ObjectiveValue) = MOI.eval_objective(model.qpblock, model.primal)
MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time
function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex) 
    MOI.check_result_index_bounds(model, attr)
    return model.primal[vi.value]
end
MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.status

MOI.supports(::Optimizer, ::MOI.Silent) = true;

function construct_admittance_matrix() 
    
end
# Now for the constraint bridge

if false
    rng = MersenneTwister(3)
    N = 10
    M = 6
    Q = linalg.I(N)
    Q2 = 0.2*linalg.diagm(1.0:N)
    Q3 = 2*linalg.diagm(1.0:N)
    b = rand(rng, M) * 10
    A = rand(rng, M, N)
    # Q = Q * Q'
    # print(Q)
    c = rand(rng, N)
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:N])
    @objective(model, MIN_SENSE, c' * x+  2 * x' * I * x)
    # @constraint(model, A * x <= b)
    # @constraint(model, x>=0)

    # @constraint(model, x[1] <= 1)
    # @constraint(model, x[1] >= 0.2)
    @constraint(model, 2 <= x' * Q3 * x <=20)
    optimize!(model)
    println(model)
    model2 = Model(Optimizer)
    @variable(model2, x[1:N])
    @objective(model2, MIN_SENSE,  c' * x + 2 * x' * I * x)
    # @constraint(model2, A * x <= b)

    @constraint(model2, 2 <= x' * Q3 * x <=20)
    # @constraint(model2, x>=0)
    optimize!(model2)
elseif true
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
    power_file = ENV["PGLIB"] * "/pglib_opf_case2_lmbd.m";
    data = parse_file(power_file)
    Y = calc_admittance_matrix(data).matrix
    YR = to_real_rep(Y)
    YI = to_real_rep(-1im * Y)
    # permute = 
    N = size(Y)[1]
    G = real.(Y)
    B = imag.(Y)

    n = Y.m;
    C = zeros(4,4)
    C[1,1] = data["gen"]["1"]["cost"][2]
    C[3,3] = data["gen"]["1"]["cost"][2]
    C[2,2] = data["gen"]["2"]["cost"][2]
    C[4,4] = data["gen"]["2"]["cost"][2]
    PU = data["gen"]["1"]
    test_model = Model(Ipopt.Optimizer)
    @variable(test_model, U[1:2N])
    @objective(test_model, MIN_SENSE, U' * C * U)
    # eⱼ = zeros(N); eⱼ[j] = 1; Eⱼ = diagm(eⱼ);
    for j in 1:N
        eⱼ = zeros(N); eⱼ[j] = 1; Eⱼ = diagm(eⱼ);Ψⱼ = to_real_rep(Eⱼ * Y);Φⱼ = to_real_rep(-im * Eⱼ * Y);
        @constraint(test_model, data["gen"]["$(j)"]["pmin"]-data["load"]["$(j)"]["pd"] <= U' * Ψⱼ * U <= data["gen"]["$(j)"]["pmax"]-data["load"]["$(j)"]["pd"])
        @constraint(test_model, data["gen"]["$(j)"]["qmin"]+data["load"]["$(j)"]["qd"]  <= U' * Φⱼ * U <= data["gen"]["$(j)"]["qmax"]+data["load"]["$(j)"]["qd"])
        @constraint(test_model, 0.80 <= U[j]^2 + U[j+N]^2 <= 1.23)
    end
    # @constraint(test_model, U <= 1.1 * ones(2N))
    # @constraint(test_model, U >= 0.9 * ones(2N))

    # @constraint(test_model, )
    pm = instantiate_model(power_file, ACRPowerModel, PowerModels.build_opf);
    model = pm.model.moi_backend
    # for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
    #     if S <: MOI.EqualTo{Float64}
    #         for cind in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
    #             MOI.delete(model, cind)
    #         end
    #     end

    # end
    # optimize_model!(pm, optimizer=Optimizer)
    # optimize_model!(pm, optimizer=Ipopt.Optimizer)
else
    Y = hcat(
        [0, 1 / (0.042 + 0.9im), 1 / (0.065 + 0.62im)],
        [ 1 / (0.042 + 0.9im), 0, 1 / (0.025 + 0.75im)],
        [1 / (0.042 + 0.9im), 1 / (0.025 + 0.75im),0],
    )
    b =  hcat(
        [0, 0.3, 0.45],
        [ 0, 0, 0.7],
        [0, 0,0],
    )
    G = real(Y)
    B = imag(Y)
    Y[1, 1] += 1 / (.45im)
    Y[3, 3] += 1 / (.45im)

    model = Model(Ipopt.Optimizer)
    # @variable(model, p_g[1:3])
    # @variable(model, q_g[1:3])
    @variable(model, vd[1:3])
    @variable(model, vq[1:3])
    # @variable(model, p_12)
    # @variable(model, p_21)
    # @variable(model, p_13)
    # @variable(model, p_31)
    # @variable(model, p_23)
    # @variable(model, p_32)
    # @variable(model, q_12)
    # @variable(model, q_21)
    # @variable(model, q_13)
    # @variable(model, q_31)
    # @variable(model, q_23)
    # @variable(model, q_32)
    @objective(model, MIN_SENSE, 1100 * vd[1]^2*vq[1]^2 + 500 * vd[1]*vq[1] + 850 * p_g[2]^2 + 120 * p_g[2])

    @constraint(model,0 <= p_g[1] <= 20)
    @constraint(model,0 <= p_g[2] <= 20)
    @constraint(model,0 <= p_g[3] <= 0)
    @constraint(model,-10 <= q_g[1] <= 10)
    @constraint(model,-10 <= q_g[2] <= 10)
    @constraint(model,-10 <= q_g[3] <= 10)
    @constraint(model,0.9^2 <= vd[1]^2 + vq[1]^2 <= 1.1^2)
    @constraint(model,0.9^2 <= vd[2]^2 + vq[2]^2 <= 1.1^2)
    @constraint(model,0.9^2 <= vd[3]^2 + vq[3]^2 <= 1.1^2)

    @constraint(model,p_12^2 + q_12^2 <= 8100)
    @constraint(model,p_21^2 + q_21^2 <= 8100)

    @constraint(model,p_13^2 + q_13^2 <= 8100)
    @constraint(model,p_31^2 + q_31^2 <= 8100)

    @constraint(model,p_23^2 + q_23^2 <= 0.25)
    @constraint(model,p_32^2 + q_32^2 <= 0.25)
    @constraint(model,-1.1 <= vd[1] <= 1.1)
    @constraint(model,-1.1 <= vd[2] <= 1.1)
    @constraint(model,-1.1 <= vd[3] <= 1.1)
    @constraint(model,-1.1 <= vq[1] <= 1.1)
    @constraint(model,-1.1 <= vq[2] <= 1.1)
    @constraint(model,-1.1 <= vq[3] <= 1.1)
    @constraint(model,p_12 + p_13 + p_g[1]-1.10 == -vd[1] * (
        vd[2] * G[1,2] - vq[2] * B[1,2]+vd[3] * G[1,3] - vq[3] * B[1,3]))
    @constraint(model,p_g[2]-110 == -vd[2] * (
        vd[1] * G[2,1] - vq[1] * B[2,1]+vd[3] * G[2,3] - vq[3] * B[2,3]))
    @constraint(model,p_g[1]-95 == -vd[2] * (
        vd[1] * G[2,1] - vq[1] * B[2,1]+vd[3] * G[2,3] - vq[3] * B[2,3]))
    print(model)
    # @constraint(model,p_g[1]-110 == vd[1] * (1/))

    # pm = instantiate_model(power_file, ACRPowerModel, PowerModels.build_opf)
    # model = pm.model.moi_backend
    # println(pm.model)

    # optimize_model!(pm, optimizer=Optimizer)
    # optimize_model!(pm, optimizer=Ipopt.Optimizer)
end
# optimize_model!(pm,)
# moi_model = pm.model.moi_backend;

# data = parse_file(ENV["PGLIB"] * "/pglib_opf_case3_lmbd.m")
# model_1 = solve_ac_opf(data, Ipopt.Optimizer)
# model_2 = solve_ac_opf(data, Optimizer)

# println(solution_summary(model2))
# function MOI.optimize!(model::Optimizer)
    
#     end
#     SetIntermediateCallback(inner, _moi_callback)
#     IpoptSolve(inner)
#     model.solve_time = time() - start_time
#     return
# end