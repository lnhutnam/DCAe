module DifferenceOfConvex

using SparseArrays, LinearAlgebra, Statistics, Random, Roots
using Plots

# Export main functions
export DCA, iDCA, DCAe, run_experiment, plot_results

# Include all component files
include("Utils.jl")
include("Operations.jl")
include("Operators.jl")
include("DCA.jl")
include("iDCA.jl")
include("DCAe.jl")
include("Experiments.jl")

end # module
