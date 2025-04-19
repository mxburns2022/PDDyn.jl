# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# <From Ipopt.jl>
# !!! warning
#
#     The contents of this file are experimental.
#
#     Until this message is removed, breaking changes to the functions and
#     types, including their deletion, may be introduced in any minor or patch
#     release of Ipopt.

using ForwardDiff
const _PARAMETER_OFFSET = 0x00f0000000000000

_is_parameter(x::MOI.VariableIndex) = x.value >= _PARAMETER_OFFSET

_is_parameter(term::MOI.ScalarAffineTerm) = _is_parameter(term.variable)

function _is_parameter(term::MOI.ScalarQuadraticTerm)
    return _is_parameter(term.variable_1) || _is_parameter(term.variable_2)
end

@enum(
    _FunctionType,
    _kFunctionTypeVariableIndex,
    _kFunctionTypeScalarAffine,
    _kFunctionTypeScalarQuadratic,
)

function _function_type_to_func(::Type{Float64}, k::_FunctionType)
    if k == _kFunctionTypeVariableIndex
        return MOI.VariableIndex
    elseif k == _kFunctionTypeScalarAffine
        return MOI.ScalarAffineFunction{Float64}
    else
        @assert k == _kFunctionTypeScalarQuadratic
        return MOI.ScalarQuadraticFunction{Float64}
    end
end

_function_info(::MOI.VariableIndex) = _kFunctionTypeVariableIndex
_function_info(::MOI.ScalarAffineFunction) = _kFunctionTypeScalarAffine
_function_info(::MOI.ScalarQuadraticFunction) = _kFunctionTypeScalarQuadratic

@enum(
    _BoundType,
    _kBoundTypeLessThan,
    _kBoundTypeGreaterThan,
    _kBoundTypeEqualTo,
    _kBoundTypeInterval,
)

_set_info(s::MOI.LessThan) = _kBoundTypeLessThan, -Inf, s.upper, 1, -s.upper
_set_info(s::MOI.GreaterThan) = _kBoundTypeGreaterThan, s.lower, Inf, -1, s.lower
_set_info(s::MOI.EqualTo) = _kBoundTypeEqualTo, s.value, s.value, 1, 1
_set_info(s::MOI.Interval) = _kBoundTypeInterval, s.lower, s.upper, 1, 1

function _bound_type_to_set(::Type{Float64}, k::_BoundType)
    if k == _kBoundTypeEqualTo
        return MOI.EqualTo{Float64}
    elseif k == _kBoundTypeLessThan
        return MOI.LessThan{Float64}
    elseif k == _kBoundTypeGreaterThan
        return MOI.GreaterThan{Float64}
    else
        @assert k == _kBoundTypeInterval
        return MOI.Interval{Float64}
    end
end

function max_coeff(f::MOI.ScalarAffineFunction{Float64})
    maxfound = abs(f.constant)
    for t in f.terms
        maxfound = max(maxfound, abs(t.coefficient))
    end
    return maxfound
end

function max_coeff(f::MOI.ScalarQuadraticFunction{Float64})
    maxfound = abs(f.constant)
    for t in f.affine_terms
        maxfound = max(maxfound, abs(t.coefficient))
    end
    for t in f.quadratic_terms
        maxfound = max(maxfound, abs(t.coefficient))
    end
    return maxfound
end
function max_coeff(::MOI.VariableIndex)
    return 1.0
end
mutable struct QPBlockData{Float64}
    objective::Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}
    objective_function_type::_FunctionType
    constraints::Vector{
        Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    }
    g_L::Vector{Float64}
    g_U::Vector{Float64}
    mult_g::Vector{Union{Nothing,Float64}}
    function_type::Vector{_FunctionType}
    bound_type::Vector{_BoundType}
    parameters::Dict{Int64,Float64}
    norm_constant::Float64

    function QPBlockData{Float64}()
        return new(
            zero(MOI.ScalarQuadraticFunction{Float64}),
            _kFunctionTypeScalarAffine,
            Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}[],
            Float64[],
            Float64[],
            Union{Nothing,Float64}[],
            _FunctionType[],
            _BoundType[],
            Dict{Int64,Float64}(),
            1.
        )
    end


end

function MOI.empty!(block::QPBlockData{Float64})
    empty!(block.constraints)
    empty!(block.g_L)
    empty!(block.g_U)
    empty!(block.mult_g)
    empty!(block.function_type)
    empty!(block.bound_type)
    empty!(block.parameters)
end

function MOI.is_empty(block::QPBlockData{Float64})
    return size(block.constraints)[1] == 0
end

function _value(variable::MOI.VariableIndex, x::Vector, p::Dict)
    if _is_parameter(variable)
        return p[variable.value]
    else
        return x[variable.value]
    end
end

function eval_function(
    f::MOI.ScalarQuadraticFunction{Float64},
    x::AbstractVector,
    p::Dict{Int64,Float64},
)::Float64
    y = f.constant
    for term in f.affine_terms
        y += ForwardDiff.value(term.coefficient * _value(term.variable, x, p))
    end
    for term in f.quadratic_terms
        v1 = _value(term.variable_1, x, p)
        v2 = _value(term.variable_2, x, p)
        if term.variable_1 == term.variable_2
            y += ForwardDiff.value(term.coefficient * v1 * v2 / 2)
        else
            y += ForwardDiff.value(term.coefficient * v1 * v2)
        end
    end
    return y
end

function eval_function(
    f::MOI.ScalarAffineFunction{Float64},
    x::Vector{Float64},
    p::Dict{Int64,Float64},
)::Float64
    y = f.constant
    for term in f.terms
        y += term.coefficient * _value(term.variable, x, p)
    end
    return y
end

function eval_dense_gradient(
    ∇f::Vector{Float64},
    f::MOI.ScalarQuadraticFunction{Float64},
    x::Vector{Float64},
    p::Dict{Int64,Float64},
)::Nothing
    for term in f.affine_terms
        if !_is_parameter(term.variable)
            ∇f[term.variable.value] += term.coefficient
        end
    end
    for term in f.quadratic_terms
        if !_is_parameter(term.variable_1)
            v = _value(term.variable_2, x, p)
            ∇f[term.variable_1.value] += term.coefficient * v
        end
        if term.variable_1 != term.variable_2 && !_is_parameter(term.variable_2)
            v = _value(term.variable_1, x, p)
            ∇f[term.variable_2.value] += term.coefficient * v
        end
    end
    return
end

function eval_dense_gradient(
    ∇f::Vector{Float64},
    f::MOI.ScalarQuadraticFunction{Float64},
    x::Vector{ForwardDiff.Dual{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, Float64}, Float64, 1}},
    p::Dict{Int64,Float64},
)::Nothing
    for term in f.affine_terms
        if !_is_parameter(term.variable)
            ∇f[term.variable.value] += term.coefficient
        end
    end
    for term in f.quadratic_terms
        if !_is_parameter(term.variable_1)
            v = ForwardDiff.value(_value(term.variable_2, x, p))
            ∇f[term.variable_1.value] += term.coefficient * v
        end
        if term.variable_1 != term.variable_2 && !_is_parameter(term.variable_2)
            v = ForwardDiff.value(_value(term.variable_1, x, p))
            ∇f[term.variable_2.value] += term.coefficient * v
        end
    end
    return
end

function eval_dense_gradient(
    ∇f::Vector{Float64},
    f::MOI.ScalarAffineFunction{Float64},
    x::Vector{ForwardDiff.Dual{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, Float64}, Float64, 1}},
    p::Dict{Int64,Float64},
)::Nothing
    for term in f.terms
        if !_is_parameter(term.variable)
            ∇f[term.variable.value] += term.coefficient
        end
    end
   
    return
end
function normalize(block::QPBlockData) 
    block.norm_constant = max_coeff(block.objective)
    for c in block.constraints
        block.norm_constant = max(block.norm_constant, max_coeff(c))
    end
end

function eval_dense_gradient(
    ∇f::Vector,
    f::MOI.ScalarAffineFunction{Float64},
    x::Vector{Float64},
    p::Dict{Int64,Float64},
)::Nothing
    for term in f.terms
        if !_is_parameter(term.variable)
            ∇f[term.variable.value] += term.coefficient
        end
    end
    return
end

function append_sparse_gradient_structure!(
    f::MOI.ScalarQuadraticFunction,
    J,
    row,
)
    for term in f.affine_terms
        if !_is_parameter(term.variable)
            push!(J, (row, term.variable.value))
        end
    end
    for term in f.quadratic_terms
        if !_is_parameter(term.variable_1)
            push!(J, (row, term.variable_1.value))
        end
        if term.variable_1 != term.variable_2 && !_is_parameter(term.variable_2)
            push!(J, (row, term.variable_2.value))
        end
    end
    return
end

function append_sparse_gradient_structure!(f::MOI.ScalarAffineFunction, J, row)
    for term in f.terms
        if !_is_parameter(term.variable)
            push!(J, (row, term.variable.value))
        end
    end
    return
end

function  eval_sparse_gradient(
    ∇f::AbstractVector,
    f::MOI.ScalarQuadraticFunction{Float64},
    x::Vector,
    p::Dict{Int64,Float64},
)::Int
    i = 0
    for term in f.affine_terms
        if !_is_parameter(term.variable)
            i += 1
            ∇f[i] = term.coefficient
        end
    end
    for term in f.quadratic_terms
        if !_is_parameter(term.variable_1)
            v = _value(term.variable_2, x, p)
            i += 1
            ∇f[i] = term.coefficient * v
        end
        if term.variable_1 != term.variable_2 && !_is_parameter(term.variable_2)
            v = _value(term.variable_1, x, p)
            i += 1
            ∇f[i] = term.coefficient * v
        end
    end
    return i
end
function  eval_sparse_gradient(
    ∇f::AbstractVector,
    f::MOI.ScalarQuadraticFunction{Float64},
    x::Vector{ForwardDiff.Dual{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, Float64}, Float64, 1}},
    p::Dict{Int64,Float64},
)::Int
    i = 0
    for term in f.affine_terms
        if !_is_parameter(term.variable)
            i += 1
            ∇f[i] = term.coefficient
        end
    end
    for term in f.quadratic_terms
        if !_is_parameter(term.variable_1)
            v = ForwardDiff.value(_value(term.variable_2, x, p))
            i += 1
            ∇f[i] = term.coefficient * v
        end
        if term.variable_1 != term.variable_2 && !_is_parameter(term.variable_2)
            v = ForwardDiff.value(_value(term.variable_1, x, p))
            i += 1
            ∇f[i] = term.coefficient * v
        end
    end
    return i
end

function eval_sparse_gradient(
    ∇f::AbstractVector,
    f::MOI.ScalarAffineFunction{Float64},
    x::Vector,
    p::Dict{Int64,Float64},
)::Int
    i = 0
    for term in f.terms
        if !_is_parameter(term.variable)
            i += 1
            ∇f[i] = term.coefficient
        end
    end
    return i
end

function append_sparse_hessian_structure!(f::MOI.ScalarQuadraticFunction, H)
    for term in f.quadratic_terms
        if _is_parameter(term.variable_1) || _is_parameter(term.variable_2)
            continue
        end
        push!(H, (term.variable_1.value, term.variable_2.value))
    end
    return
end

append_sparse_hessian_structure!(::MOI.ScalarAffineFunction, H) = nothing

function eval_sparse_hessian(
    ∇²f::AbstractVector{Float64},
    f::MOI.ScalarQuadraticFunction{Float64},
    σ::Float64,
)::Int
    i = 0
    for term in f.quadratic_terms
        if _is_parameter(term.variable_1) || _is_parameter(term.variable_2)
            continue
        end
        i += 1
        ∇²f[i] = term.coefficient * σ
    end
    return i
end

function eval_sparse_hessian(
    ∇²f::AbstractVector{Float64},
    f::MOI.ScalarAffineFunction{Float64},
    σ::Float64,
)::Int
    return 0
end

Base.length(block::QPBlockData) = length(block.bound_type)

function MOI.set(
    block::QPBlockData{Float64},
    ::MOI.ObjectiveFunction{F},
    f::F,
) where {Float64,F<:Union{MOI.VariableIndex,MOI.ScalarAffineFunction{Float64}}}
    block.objective = convert(MOI.ScalarAffineFunction{Float64}, f)
    block.objective_function_type = _function_info(f)
    return
end

function MOI.set(
    block::QPBlockData{Float64},
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
    f::MOI.ScalarQuadraticFunction{Float64},
)
    block.objective = f
    block.objective_function_type = _function_info(f)
    return
end

function MOI.get(block::QPBlockData{Float64}, ::MOI.ObjectiveFunctionType)
    return _function_type_to_func(Float64, block.objective_function_type)
end

function MOI.get(block::QPBlockData{Float64}, ::MOI.ObjectiveFunction{F}) where {Float64,F}
    return convert(F, block.objective)
end

function MOI.get(
    block::QPBlockData{Float64},
    ::MOI.ListOfConstraintTypesPresent,
)
    constraints = Set{Tuple{Type,Type}}()
    for i in 1:length(block)
        F = _function_type_to_func(Float64, block.function_type[i])
        S = _bound_type_to_set(Float64, block.bound_type[i])
        push!(constraints, (F, S))
    end
    return collect(constraints)
end

function MOI.is_valid(
    block::QPBlockData{Float64},
    ci::MOI.ConstraintIndex{F,S},
) where {
    Float64,
    F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    S<:Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}},
}
    return 1 <= ci.value <= length(block)
end

function MOI.get(
    block::QPBlockData{Float64},
    ::MOI.ListOfConstraintIndices{F,S},
) where {
    Float64,
    F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    S<:Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}},
}
    ret = MOI.ConstraintIndex{F,S}[]
    for i in 1:length(block)
        if _bound_type_to_set(Float64, block.bound_type[i]) != S
            continue
        elseif _function_type_to_func(Float64, block.function_type[i]) != F
            continue
        end
        push!(ret, MOI.ConstraintIndex{F,S}(i))
    end
    return ret
end

function MOI.get(
    block::QPBlockData{Float64},
    ::MOI.NumberOfConstraints{F,S},
) where {
    Float64,
    F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    S<:Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64},MOI.Interval{Float64}},
}
    return length(MOI.get(block, MOI.ListOfConstraintIndices{F,S}()))
end

function MOI.add_constraint(
    block::QPBlockData{Float64},
    f::Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    s::MOI.EqualTo{Float64},
)
    g = f - s.value
    push!(block.constraints, g)
    push!(block.g_L, s.value)
    push!(block.g_U, s.value)
    push!(block.mult_g, nothing)
    push!(block.bound_type, _kBoundTypeLessThan)
    push!(block.function_type, _function_info(f))
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(length(block.bound_type))
end

function MOI.add_constraint(
    block::QPBlockData{Float64},
    f::Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    s::MOI.LessThan{Float64}
)
    g = f - s.upper
    push!(block.constraints, g)
    push!(block.g_L, -Inf)
    push!(block.g_U, s.upper)
    push!(block.mult_g, nothing)
    push!(block.bound_type, _kBoundTypeLessThan)
    push!(block.function_type, _function_info(f))
    return MOI.ConstraintIndex{typeof(f),MOI.LessThan{Float64}}(length(block.bound_type))
end

function MOI.add_constraint(
    block::QPBlockData{Float64},
    f::Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    s::MOI.GreaterThan{Float64},
)
    
    g = -f + s.lower
    push!(block.constraints, g)
    push!(block.g_L,s.lower)
    push!(block.g_U, Inf)
    push!(block.mult_g, nothing)
    push!(block.bound_type, _kBoundTypeLessThan)
    push!(block.function_type, _function_info(f))
    return MOI.ConstraintIndex{typeof(f),MOI.LessThan{Float64}}(length(block.bound_type))
end


function MOI.get(
    block::QPBlockData{Float64},
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{F,S},
) where {Float64,F,S}
    return convert(F, block.constraints[c.value])
end

function MOI.get(
    block::QPBlockData{Float64},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,S},
) where {Float64,F,S}
    row = c.value
    if block.bound_type[row] == _kBoundTypeEqualTo
        return MOI.EqualTo(block.g_L[row])
    elseif block.bound_type[row] == _kBoundTypeLessThan
        return MOI.LessThan(block.g_U[row])
    elseif block.bound_type[row] == _kBoundTypeGreaterThan
        return MOI.GreaterThan(block.g_L[row])
    else
        @assert block.bound_type[row] == _kBoundTypeInterval
        return MOI.Interval(block.g_L[row], block.g_U[row])
    end
end

function MOI.set(
    block::QPBlockData{Float64},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,MOI.LessThan{Float64}},
    set::MOI.LessThan{Float64},
) where {Float64,F}
    row = c.value
    block.g_U[row] = set.upper
    return
end

function MOI.set(
    block::QPBlockData{Float64},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,MOI.GreaterThan{Float64}},
    set::MOI.GreaterThan{Float64},
) where {Float64,F}
    row = c.value
    block.g_L[row] = set.lower
    return
end

function MOI.set(
    block::QPBlockData{Float64},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,MOI.EqualTo{Float64}},
    set::MOI.EqualTo{Float64},
) where {Float64,F}
    row = c.value
    block.g_L[row] = set.value
    block.g_U[row] = set.value
    return
end

function MOI.set(
    block::QPBlockData{Float64},
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{F,MOI.Interval{Float64}},
    set::MOI.Interval{Float64},
) where {Float64,F}
    row = c.value
    block.g_L[row] = set.lower
    block.g_U[row] = set.upper
    return
end

function MOI.get(
    block::QPBlockData{Float64},
    ::MOI.ConstraintDualStart,
    c::MOI.ConstraintIndex{F,S},
) where {Float64,F,S}
    return block.mult_g[c.value]
end

function MOI.set(
    block::QPBlockData{Float64},
    ::MOI.ConstraintDualStart,
    c::MOI.ConstraintIndex{F,S},
    value,
) where {Float64,F,S}
    block.mult_g[c.value] = value
    return
end

function MOI.eval_objective(
    block::QPBlockData{Float64},
    x::AbstractVector{Float64},
)
    return eval_function(block.objective, x, block.parameters)
end

function MOI.eval_objective_gradient(
    block::QPBlockData{Float64},
    ∇f::AbstractVector,
    x::AbstractVector,
)
    ∇f .= zero(Float64)
    eval_dense_gradient(∇f, block.objective, x, block.parameters)
    return
end

function MOI.eval_constraint(
    block::QPBlockData{Float64},
    g::AbstractVector,
    x::AbstractVector,
)
    for (i, constraint) in enumerate(block.constraints)
        g[i] = eval_function(constraint, x, block.parameters)
    end
    return
end

function MOI.jacobian_structure(block::QPBlockData)
    J = Tuple{Int,Int}[]
    for (row, constraint) in enumerate(block.constraints)
        append_sparse_gradient_structure!(constraint, J, row)
    end
    return J
end

function MOI.eval_constraint_jacobian(
    block::QPBlockData{Float64},
    J::AbstractVector{Float64},
    x::AbstractVector,
)
    i = 1
    for constraint in block.constraints
        ∇f = view(J, i:length(J))
        i += eval_sparse_gradient(∇f, constraint, x, block.parameters)
    end
    return i
end


function MOI.hessian_lagrangian_structure(block::QPBlockData)
    H = Tuple{Int,Int}[]
    append_sparse_hessian_structure!(block.objective, H)
    for constraint in block.constraints
        append_sparse_hessian_structure!(constraint, H)
    end
    return H
end

function MOI.eval_hessian_lagrangian(
    block::QPBlockData{Float64},
    H::AbstractVector{Float64},
    x::AbstractVector{Float64},
    σ::Float64,
    μ::AbstractVector{Float64},
)
    i = 1
    i += eval_sparse_hessian(H, block.objective, σ)
    for (row, constraint) in enumerate(block.constraints)
        ∇²f = view(H, i:length(H))
        i += eval_sparse_hessian(∇²f, constraint, μ[row])
    end
    return i
end
