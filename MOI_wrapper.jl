include("PDDyn.jl")


mutable struct Optimizer <: MOI.AbstractOptimizer
    # Structures storing problem data
    dim::Int
    qpblock::QPBlockData{Float64}

    # Integrator information
    τ::Float64
    iterations::Int

    # Problem state
    primal::Vector{Float64}
    dual::Vector{Float64}
    solve_time::Float64
    obj_value::Float64
    status::MOI.TerminationStatusCode
    dual_status::MOI.DualStatus
    primal_status::MOI.PrimalStatus
    rcount::MOI.ResultCount
    silent::Bool;

    function Optimizer() 
        return new(
            0,
            QPBlockData{Float64}(),
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
    return isempty(model.qpblock) && model.status == MOI.OPTIMIZE_NOT_CALLED
end
function MOI.empty!(model::Optimizer)
    empty!(model.qpblock)
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
    if model.status == MOI.LOCALLY_OPTIMAL
        return "found a locally optimal primal-dual solution within provided tolerances"
    end
    return "failed to solve"
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if model.status == MOI.LOCALLY_OPTIMAL && model.primal_status == MOI.FEASIBLE_POINT && model.dual_status == MOI.FEASIBLE_POINT
        return 1
    end
    return 0
end

function optimize!(model::Optimizer, src::MOI.ModelLike) 
    
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

