# PDDyn.jl
The primal dual gradient dynamics (PDGD) simulator code can be found in PDDyn.jl, with JuMP/MathOptInterface utilities in MOI_wrapper.jl and Utilities.jl. 
Note that a large portion of the functionality in Utilities.jl was adapted from the public [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) repository.

QCQP experiment code can be found in [qcqp_experiment.jl](qcqp_experiment.jl) and ACOPF experiment code can be found in [acopf_experiment.jl](acopf_experiment.jl). 
The QCQP experiments use code from the local [instances](instances) folder, which contains problems adapted from [MINLPLib.jl](https://github.com/lanl-ansi/MINLPLib.jl). 
ACOPF experiments attempt to load files from the [pglib-opf](https://github.com/power-grid-lib/pglib-opf) repository using the PGLIB environment variable.

Code to generate plots can be found in [visualization.ipynb](visualization.ipynb).
