include("PDDyn.jl")
import LinearAlgebra as linalg

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
    ::Type{<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction},
    ::Type{<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction},
    ::Type{<:Union{MOI.LessThan,MOI.GreaterThan}},
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
        constraintset::Union{MOI.EqualTo, MOI.LessThan, MOI.GreaterThan})
    MOI.add_constraint(model.qpblock, cons, constraintset)
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
    cmodel = CacheModel()
    index_map = MOI.copy_to(cmodel, src)
    for (j, k) in index_map
        dest.x_to_index[j] = k.value
    end
    ftype = MOI.get(src, MOI.ObjectiveFunctionType())
    objfunc = MOI.get(src, MOI.ObjectiveFunction{ftype}())
    MOI.set(dest.qpblock, MOI.ObjectiveFunction{ftype}(), objfunc)
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
            func = MOI.get(src, MOI.ConstraintFunction(), ci)
            MOI.add_constraint(dest.qpblock, func, set)
        end
    end
    N = MOI.get(src, MOI.NumberOfVariables())

    dest.status, dest.primal_status, dest.dual_status, dest.solve_time, dest.primal, dest.dual = solve_pddyn(dest.qpblock, N, verbose=!dest.silent)
    dest.obj_value = MOI.eval_objective(dest.qpblock, dest.primal)

    # dest.dual_status = MOI.DualStatus
    # for src_ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    #     dest.ci_to_rows[index_map[src_ci]] = MOI.Utilities.rows(cache.constraints.sets, index_map[src_ci])
    # end
    # for (i, src_x) in enumerate(MOI.get(src, MOI.ListOfVariableIndices()))
    #     dest.x_to_col[index_map[src_x]] = i
    # end
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


Q = linalg.I(10)
# Q = Q * Q'
# print(Q)
c = rand(10)
model = Model(Optimizer)
@variable(model, x[1:10])
@objective(model, MIN_SENSE, x' * Q * x + c' * x)
set_silent(model)
optimize!(model)
println(solution_summary(model))
model2 = Model(Ipopt.Optimizer)
@variable(model2, y[1:10])
@objective(model2, MIN_SENSE, y' * Q * y + c' * y)
set_silent(model2)
optimize!(model2)
println(solution_summary(model2))
# function MOI.optimize!(model::Optimizer)
    
#     end
#     SetIntermediateCallback(inner, _moi_callback)
#     IpoptSolve(inner)
#     model.solve_time = time() - start_time
#     return
# end