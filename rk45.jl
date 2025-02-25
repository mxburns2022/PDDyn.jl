import LinearAlgebra


mutable struct RK45Integrator
    scratch::Vector{Vector{Float64}} # Scratchpad memories
    stepsize::Float64

    function RK45Integrator(dimension::Int, stepsize::Float64) 
        return new([fill(0.0, dimension) for _ in 1:4], stepsize)
    end
end

function rks_step!(integrator::RK45Integrator, state::Vector{Float64}, gradient_oracle::Function, time::Float64)
    η = integrator.stepsize
    gradient_oracle(integrator.scratch[1], state, time)
    gradient_oracle(integrator.scratch[2], state + 0.5 * η * integrator.scratch[1], time + 0.5 * η)
    gradient_oracle(integrator.scratch[3], state + 0.5 * η * integrator.scratch[2], time + 0.5 * η)
    gradient_oracle(integrator.scratch[4], state + η * integrator.scratch[4], time + η)
    LinearAlgebra.axpy!(1.0, integrator.scratch[4], integrator.scratch[1])
    LinearAlgebra.axpy!(2.0, integrator.scratch[3], integrator.scratch[1])
    LinearAlgebra.axpy!(2.0, integrator.scratch[2], integrator.scratch[1])
    LinearAlgebra.axpy!(η / 6.0, integrator.scratch[1], state)
end
