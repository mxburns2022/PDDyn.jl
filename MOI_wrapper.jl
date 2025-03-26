include("PDDyn.jl")
using PowerModels
import LinearAlgebra as linalg
import MathOptInterface.Utilities as MOIU
import SparseArrays as sp
using Ipopt
# using MosekTools
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
    # println("N", N)
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
                                                                                                             τ=1e-5,
                                                                                                             tstop=10000.,
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
