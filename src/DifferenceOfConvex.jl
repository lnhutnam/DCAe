module DifferenceOfConvex

using SparseArrays, LinearAlgebra, Statistics, Random, Roots
using Plots

# Export main functions AND utility functions
export DCA, iDCA, DCAe, run_experiment, plot_results, analyze_results
export power_method, obj_func, MatCompRMSE

# Include all component files
include("Utils.jl")
include("Operations.jl")
include("Operators.jl")
include("DCA.jl")
include("iDCA.jl")
include("DCAe.jl")
include("Experiments.jl")

end # module