# main.jl - Example script for using DifferenceOfConvex package

using SparseArrays, LinearAlgebra, Statistics, Random
using JLD, JLD2  # For loading/saving data
using Plots  # For visualization

# Add the package directory to the path if needed
# push!(LOAD_PATH, pwd())

# Import our module
include("DifferenceOfConvex.jl")
using .DifferenceOfConvex

"""
    generate_synthetic_data(m, n, r, density=0.1)
    
Generate synthetic matrix completion data.

# Arguments
- `m`, `n`: Matrix dimensions
- `r`: True rank of the matrix
- `density`: Fraction of observed entries

# Returns
- Sparse matrix of observations
- True U matrix
- True V matrix
"""
function generate_synthetic_data(m, n, r, density=0.1)
    # Create low-rank matrices
    U_true = randn(m, r)
    V_true = randn(n, r)
    
    # Generate full matrix
    full_matrix = U_true * V_true'
    
    # Sample entries to create sparse observation matrix
    nnz = floor(Int, m * n * density)
    indices = randperm(m * n)[1:nnz]
    
    i_indices = [div(idx-1, n) + 1 for idx in indices]
    j_indices = [mod(idx-1, n) + 1 for idx in indices]
    
    # Add some noise
    values = [full_matrix[i, j] + 0.1 * randn() for (i, j) in zip(i_indices, j_indices)]
    
    # Create sparse matrix
    sparse_matrix = sparse(i_indices, j_indices, values, m, n)
    
    return sparse_matrix, U_true, V_true
end

"""
    run_synthetic_example()
    
Run a demonstration using synthetic data.
"""
function run_synthetic_example()
    println("=== Running synthetic data example ===")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Generate synthetic data
    println("Generating synthetic data...")
    m, n, r = 500, 300, 5  # Matrix dimensions and rank
    density = 0.1  # 10% of entries observed
    data_matrix, U_true, V_true = generate_synthetic_data(m, n, r, density)
    
    # Save synthetic data for future use
    save("synthetic_data.jld2", "data", data_matrix, "U_true", U_true, "V_true", V_true)
    println("Synthetic data saved to synthetic_data.jld2")
    
    # Set parameters for algorithms
    params = Dict(
        :maxR => r,       # Rank of approximation
        :maxtime => 10,   # Maximum runtime (seconds)
        :maxIter => 50,   # Maximum iterations
        :tol => 1e-6,     # Convergence tolerance
        :fun_num => 4,    # Regularization function (4 = exponential)
        :lambda => 0.1,   # Regularization strength
        :theta => 5       # Regularization parameter for exponential
    )
    
    # Run experiment
    results = run_experiment("synthetic_data.jld2", params)
    
    # Plot results
    objective_plot, rmse_plot, sparsity_plot = plot_results(results)
    
    # Save plots
    savefig(objective_plot, "objective_comparison.png")
    savefig(rmse_plot, "rmse_comparison.png")
    savefig(sparsity_plot, "sparsity_comparison.png")
    
    # Display plots
    display(objective_plot)
    display(rmse_plot)
    display(sparsity_plot)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print analysis
    println("\n=== Performance Analysis ===")
    for (method, metrics) in analysis
        if method != :thresholds
            println("\n$method:")
            println("  Final objective: $(metrics[:final_obj])")
            println("  Final RMSE: $(metrics[:final_rmse])")
            println("  Iterations: $(metrics[:n_iterations])")
            println("  Total time: $(metrics[:total_time]) seconds")
            println("  Avg time per iteration: $(metrics[:avg_time_per_iter]) seconds")
            println("  Final sparsity (U): $(metrics[:final_sparsity].U * 100)%")
            println("  Final sparsity (V): $(metrics[:final_sparsity].V * 100)%")
        end
    end
    
    # Compare to true matrices
    println("\n=== Comparison to True Matrices ===")
    for (method, result) in results
        recovered_matrix = result[:U] * result[:V]'
        error = norm(recovered_matrix - U_true * V_true') / norm(U_true * V_true')
        println("$method relative error: $error")
    end
    
    println("\nExperiment completed. Results saved to objective_comparison.png, rmse_comparison.png, and sparsity_comparison.png")
end

"""
    run_movielens_example()
    
Run a demonstration on the MovieLens dataset if available.
"""
function run_movielens_example()
    if isfile("movielens1m.jld") || isfile("movielens1m.jld2")
        println("\n=== Running MovieLens experiment ===")
        
        # Use specific parameters for MovieLens
        params = Dict(
            :maxR => 10,      # Higher rank for real data
            :maxtime => 60,   # Longer runtime
            :maxIter => 100,  # More iterations
            :tol => 1e-8,     # Tighter tolerance
            :fun_num => 4,    # Exponential regularization
            :lambda => 0.5,   # Stronger regularization
            :theta => 5       # Exponential parameter
        )
        
        # Determine file format
        file_path = isfile("movielens1m.jld") ? "movielens1m.jld" : "movielens1m.jld2"
        
        # Run experiment
        results = run_experiment(file_path, params)
        
        # Plot and save results
        objective_plot, rmse_plot, sparsity_plot = plot_results(results)
        savefig(objective_plot, "movielens_objective.png")
        savefig(rmse_plot, "movielens_rmse.png")
        savefig(sparsity_plot, "movielens_sparsity.png")
        
        # Display plots
        display(objective_plot)
        display(rmse_plot)
        display(sparsity_plot)
        
        println("\nMovieLens experiment completed. Results saved to movielens_*.png files")
    else
        println("\nMovieLens dataset not found. Skipping this example.")
        println("To run this example, place the movielens1m.jld or movielens1m.jld2 file in the current directory.")
    end
end

# Run the examples
function main()
    println("DifferenceOfConvex Algorithms - Example Runner")
    println("=============================================")
    
    # Run synthetic example
    run_synthetic_example()
    
    # Try to run MovieLens example if data is available
    run_movielens_example()
end

# Execute main function if this script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end