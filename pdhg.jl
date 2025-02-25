using JuMP
import LinearAlgebra
import MathOptInterface as MOI
import Printf
import SparseArrays
import PowerModels as PM
using Ipopt
include("rk45.jl") 





function solve_pddyn(
    Q::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Float64, Int},
    b::Vector{Float64},
    c::Vector{Float64},
    max_iters::Int = 100,
    tol::Float64 = 1e-8,
    verbose::Bool = true,
    log_freq::Int = 10_000,
    τ::Float64 = 1e-3,
    tstop::Float64 = 500.0
)
    printf(x::Float64) = Printf.@sprintf("% 1.6e", x)
    printf(x::Int) = Printf.@sprintf("%6d", x)
    # Implement primal-dual dynamics using the differential equation solver
    m, n = size(A)
    primal_dual_system = SparseArrays.spzeros(n+m, n+m)
    primal_dual_linear = zeros(n+m)
    primal_dual_linear[1:n] .= -c
    primal_dual_linear[n+1:n+m] .= -b
    primal_primal_inds = CartesianIndices((1:n, 1:n))
    dual_dual_inds = CartesianIndices((n+1:n+m, n+1:n+m))
    primal_dual_inds = CartesianIndices((1:n, n+1:n+m))
    dual_primal_inds = CartesianIndices((n+1:n+m, 1:n))
    # primal_dual_linear[n+1:n+m] .+= -d
    copyto!(primal_dual_system, primal_primal_inds, -Q, primal_primal_inds)
    copyto!(primal_dual_system, primal_dual_inds, -A', CartesianIndices(A'))
    copyto!(primal_dual_system, dual_primal_inds, A, CartesianIndices(A))
    # copyto!(primal_dual_system, dual_dual_inds, -P, CartesianIndices(P))
    # copyto!(primal_dual_system, primal_dual_inds, A', CartesianIndices(A'))
    # Projected gradient descent on Lagrangian
    #       L(x, \lambda) = 
    function grad(du, u, t)
        u[1:n] .= max.(u[1:n], 0.0)
        du .= primal_dual_system * u + primal_dual_linear
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
        x[1:n] .= max.(x[1:n], 0.0)
        # println(x)

        copy!(m_scratch, A * x[1:n])
        LinearAlgebra.axpy!(-1.0, b, m_scratch)
        pfeas = LinearAlgebra.norm(m_scratch)

        copy!(n_scratch, A' * x[n+1:n+m])
        n_scratch .+= c
        n_scratch .= min.(0.0, n_scratch)
        dfeas = LinearAlgebra.norm(n_scratch)

        copy!(n_scratch, x[1:n])
        copy!(m_scratch, x[n+1:n+m])
        # objfeas = n_scratch' * Q * n_scratch + c' * n_scratch - b' * m_scratch  
        # println(m_scratch)
        objfeas = 0.5 * n_scratch' *Q * n_scratch + c' * n_scratch - b' * m_scratch   
        if pfeas <= tol && dfeas <= tol && objfeas <= tol
            status = MOI.OPTIMAL
        elseif time >= tstop
            status = MOI.ITERATION_LIMIT
        end
        if verbose && (mod(k, log_freq) == 0 || status != MOI.OTHER_ERROR)
            logs = printf.((time, 0.5 * n_scratch' * Q * n_scratch + c' * n_scratch, -b' * m_scratch, pfeas, dfeas, objfeas))
            println(join(logs, "\t"))
        end
        time += τ   
        k += 1
    end
    return status, time, x[1:n], x[n+1:m+n]
end

# Target = LPs of the form
#=
    min cᵀx
    subject to Ax = b
    x ≥ 0

    with the Lagrangian



=#
function solve_pdhg(
        Q::SparseArrays.SparseMatrixCSC{Float64, Int},
        A::SparseArrays.SparseMatrixCSC{Float64, Int},
        b::Vector{Float64},
        c::Vector{Float64},
        max_iters::Int = 100_000,
        tol::Float64 = 1e-8,
        verbose::Bool = true,
        log_freq::Int = 1_000)
    printf(x::Float64) = Printf.@sprintf("% 1.6e", x)
    printf(x::Int) = Printf.@sprintf("%6d", x)
    m, n = size(A)
    η = τ = 1 / LinearAlgebra.norm(A) - 1e-6
    x, x_next, y, k, status = zeros(n), zeros(n), zeros(m), 0, MOI.OTHER_ERROR
    m_scratch, n_scratch = zeros(m), zeros(n)
    if verbose
        println("   iter    pobj    dobj    pfeas   dfeas   objfeas")
    end
    while status == MOI.OTHER_ERROR
        k += 1
        # Perform the primal update x_next = x - η(Aᵀy + c)
        LinearAlgebra.mul!(x_next, A', y)
        LinearAlgebra.mul!(n_scratch, Q, x)
        x_next .+= n_scratch
        LinearAlgebra.axpby!(-η, c,-η, x_next)
        x_next .+= x
        x_next .= max.(0.0, x_next)

        # Perform the dual update y_next = y + τ(A(2*x_next - x)-b)
        copy!(n_scratch, x_next)
        LinearAlgebra.axpby!(-1.0, x, 2.0, n_scratch)
        LinearAlgebra.mul!(y, A, n_scratch, τ, 1.0)
        LinearAlgebra.axpy!(-τ, b, y)

        # Copy the new iterate
        copy!(x, x_next)
        
        # Compute the primal feasibility pfeas
        LinearAlgebra.mul!(m_scratch, A, x)
        m_scratch .-= b
        pfeas = LinearAlgebra.norm(m_scratch)

        # Compute the dual feasibility dfeas
        LinearAlgebra.mul!(n_scratch, A', y)
        n_scratch .+= c
        n_scratch .= min.(0.0, n_scratch)
        dfeas = LinearAlgebra.norm(n_scratch)

        objfeas = abs(LinearAlgebra.dot(c, x) + LinearAlgebra.dot(b, y))
        if pfeas <= tol && dfeas <= tol && objfeas <= tol
            status = MOI.OPTIMAL
        elseif k == max_iters
            status = MOI.ITERATION_LIMIT
        end
        if verbose && (mod(k, log_freq) == 0 || status != MOI.OTHER_ERROR)
            logs = printf.((k, c' * x, -b' * y, pfeas, dfeas, objfeas))
            println(join(logs, "\t"))
        end
    end
    return status, k, x, y
end
"""
    Example Optimizer
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    # Map from variables to columns
    x_to_col::Dict{MOI.VariableIndex, Int}
    # Map from constraints to rows
    ci_to_rows::Dict{
        MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.Zeros},
        Vector{Int}
    }
    # Dynamical integrator information
    τ::Float64
    time::Float64
    tstop::Float64
    # Solver return information
    status::MOI.TerminationStatusCode
    iterations::Int
    x::Vector{Float64}
    y::Vector{Float64}
    solve_time::Float64
    obj_value::Float64

    function Optimizer()
        F = MOI.VectorAffineFunction{Float64}
        return new(
            Dict{MOI.VariableIndex,Int}(),
            Dict{MOI.ConstraintIndex{F,MOI.Zeros},Vector{Int}}(),
            0.0,
            0.0,
            0.0,
            MOI.OPTIMIZE_NOT_CALLED,
            0,
            Float64[],
            Float64[],
            0.0,
            0.0
        )
    end
end

function MOI.is_empty(model::Optimizer)
    # You might want to check every field, not just a few
    return isempty(model.x_to_col) && model.status == MOI.OPTIMIZE_NOT_CALLED
end

function MOI.empty!(model::Optimizer)
    empty!(model.x_to_col)
    empty!(model.ci_to_rows)
    model.status = MOI.OPTIMIZE_NOT_CALLED
    model.iterations = 0
    model.solve_time = 0.0
    model.obj_value = 0.0
    empty!(model.x)
    empty!(model.y)
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{MOI.Zeros}
)
    return true
end

MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false;

function MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Nonnegatives})
    return true;
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}
)
    return true;
end

MOI.get(::Optimizer, ::MOI.SolverName) = "PDHG"

MOI.Utilities.@product_of_sets(SetOfZeros, MOI.Zeros) # Set up the set of allowable constraint values (i.e. Ax+b ∈ {0})

# Define a GenericModel which we can customize
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
    MOI.Utilities.Quad
}

function MOI.add_constrained_variables(model::CacheModel, set::MOI.Nonnegatives)
    x = MOI.add_variables(model, MOI.dimension(set))
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives}(x[1].value)
    return x, ci
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    # Record the solve time
    start_time = time()
    # Make a problem data cache
    cache = CacheModel()
    # Copy the source model into cache using MOI Utilities
    index_map = MOI.copy_to(cache, src)
    # index_map maps from indices in src to indices in dst, allowing for dereferencing later on
    
    # ========================
    # Convert the A matrix into usable form
    A = convert(
        SparseArrays.SparseMatrixCSC{Float64, Int},
        cache.constraints.coefficients
    )

    # Copy the b vector, adding the negative sign to account for the internal representation
    b = -cache.constraints.constants

    # Objective sense makes the c vector more of a PITA, but not too bad
    sense = ifelse(cache.objective.sense == MOI.MAX_SENSE, -1, 1)
    F = MOI.ScalarAffineFunction{Float64}
    obj = MOI.get(src, MOI.ObjectiveFunction{F}())
    c = zeros(size(A, 2))
    for term in obj.terms
        c[term.variable.value] += sense * term.coefficient
    end
    # Solve the problem with PDHG and record the result
    dest.status, dest.iterations, dest.x, dest.y = solve_pdhg(A, b, c)

    # Now we need to map from the rows/cols of "dest" to the variable/constraint indices
    F, S = MOI.VectorAffineFunction{Float64}, MOI.Zeros
    for src_ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        dest.ci_to_rows[index_map[src_ci]] = MOI.Utilities.rows(cache.constraints.sets, index_map[src_ci])
    end
    for (i, src_x) in enumerate(MOI.get(src, MOI.ListOfVariableIndices()))
        dest.x_to_col[index_map[src_x]] = i
    end

    # Record the two "derived" quantities, the objective value and the solve time
    dest.obj_value = obj.constant + sense * c' * dest.x
    dest.solve_time = time() - start_time

    # NOTE: Return the index map and `false`, with the latter indicating that we do not support incremental model modification
    return index_map, false
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.status == MOI.OPTIMAL ? 1 : 0
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.status == MOI.OPTIMAL
        return "found a primal-dual optimal solution within provided tolerances"
    end
    return "failed to solve"
end

# Now implement the types of problem statuses
MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.status

function MOI.get(model::Optimizer, attr::Union{MOI.PrimalStatus, MOI.DualStatus})
    if attr.result_index == 1 && model.status == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.obj_value
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, x::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.x[model.x_to_col[x]]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros})
    MOI.check_result_index_bounds(model, attr)
    return -model.y[model.ci_to_rows[ci]]
end
MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time
MOI.get(model::Optimizer, ::MOI.BarrierIterations) = model.iterations
