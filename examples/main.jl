# Example script for using DifferenceOfConvex package

using SparseArrays, LinearAlgebra, Statistics, Random
using MAT  # For loading MAT files
using Plots  # For visualization

# Add the package directory to the path if needed
# (only necessary if the package is not installed)
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))

# Import our module
using DifferenceOfConvex

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

# Define a local power_method function to avoid module dependency issues
# function power_method(D, R, max_iter::Int, tol)
#     m, n = size(D)
#     r = size(R, 2)
#     U = randn(m, r)
    
#     for i in 1:max_iter
#         old_U = copy(U)
#         U = D * R
        
#         for j in 1:r
#             U[:, j] /= norm(U[:, j])
#         end
        
#         if norm(U - old_U) < tol
#             break
#         end
#     end
    
#     return U
# end

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
    
    # Create parameters with explicit types (to avoid type conversion errors)
    params = Dict{Symbol, Any}(
        :maxR => Int(r),       
        :maxtime => Float64(10),
        :maxIter => Int(100),
        :tol => Float64(1e-9),
        :fun_num => Int(4),
        :lambda => Float64(0.1),
        :theta => Float64(5),
        :data => "synthetic"
    )
    
    # Initialize test parameters (70% train, 30% test split)
    row, col, val = findnz(data_matrix)
    idx = randperm(length(val))
    train_idx = idx[1:floor(Int, length(val)*0.7)]
    test_idx = idx[ceil(Int, length(val)*0.3):end]
    
    # Normalize values
    val = val .- mean(val)
    val = val ./ std(val)
    
    # Create sparse train data
    train_data = sparse(row[train_idx], col[train_idx], val[train_idx], m, n)
    
    # Create test parameters with explicit array types
    test_row = row[test_idx]
    test_col = col[test_idx]
    test_data = val[test_idx]
    
    # Define R before using it
    R = randn(n, params[:maxR])
    
    # Use power_method with the newly defined R
    U0 = power_method(train_data, R, 5, 1e-6)
    
    F = svd(U0' * train_data)
    V0 = F.Vt'
    
    # Instead of using a nested dictionary, which might cause type issues,
    # we'll set up test parameters individually
    params[:test_row] = test_row
    params[:test_col] = test_col
    params[:test_data] = test_data
    params[:test_m] = m
    params[:test_n] = n
    params[:R] = R
    params[:U0] = U0
    params[:V0] = V0
    
    # Run algorithms directly
    println("\n=== Running DCA ===")
    result_dca = DCA(train_data, params[:lambda], params[:theta], params)
    
    println("\n=== Running iDCA ===")
    result_idca = iDCA(train_data, params[:lambda], params[:theta], params)
    
    println("\n=== Running DCAe ===")
    result_dcae = DCAe(train_data, params[:lambda], params[:theta], params)
    
    # Combine results
    results = Dict("DCA" => result_dca, "iDCA" => result_idca, "DCAe" => result_dcae)
    
    # Plot results
    objective_plot, rmse_plot, sparsity_plot = plot_results(results)
    
    # Save plots
    savefig(objective_plot, "synthetic_objective.png")
    savefig(rmse_plot, "synthetic_rmse.png")
    savefig(sparsity_plot, "synthetic_sparsity.png")
    
    # Display plots
    display(objective_plot)
    display(rmse_plot)
    display(sparsity_plot)
    
    # Compare to true matrices
    # println("\n=== Comparison to True Matrices ===")
    # for (method, result) in results
    #     # Extract the recovered factors
    #     recovered_U = result[:U]  # m×r matrix
    #     recovered_V = result[:V]  # n×r matrix
        
    #     # Calculate error using Frobenius norm of the difference at sampled points
    #     # This avoids constructing the full matrices
    #     error_sum = 0.0
    #     n_sampled = min(1000, length(row))  # Use a subset of points for efficiency
        
    #     # Use a random sample of points from the original data
    #     sample_indices = randperm(length(row))[1:n_sampled]
        
    #     for idx in sample_indices
    #         i = row[idx]
    #         j = col[idx]
            
    #         # Make sure indices are within bounds
    #         if i <= size(recovered_U, 1) && j <= size(recovered_V, 1)
    #             # Calculate the true value and predicted value at this point
    #             true_val = U_true[i, :] ⋅ V_true[j, :]
    #             pred_val = recovered_U[i, :] ⋅ recovered_V[j, :]
                
    #             error_sum += (true_val - pred_val)^2
    #         end
    #     end
        
    #     # Calculate relative error
    #     rmse = sqrt(error_sum / n_sampled)
        
    #     # We can't compute full matrix norm efficiently, so just report RMSE
    #     println("$method RMSE on sampled points: $rmse")
    # end
end

"""
    run_movielens_example()
    
Run a demonstration on the MovieLens dataset using the movielens1m.mat file.
"""
function run_movielens_example()
    data_path = joinpath(dirname(@__DIR__), "data", "movielens1m.mat")
    
    if isfile(data_path)
        println("\n=== Running MovieLens experiment ===")
        
        # Load the MAT file
        println("Loading MovieLens1M dataset...")
        matdata = matread(data_path)
        data = matdata["data"]  # Extract the data matrix
        
        # Create parameters with explicit types
        params = Dict{Symbol, Any}(
            :maxR => Int(5),
            :maxtime => Float64(20),
            :maxIter => Int(50),
            :tol => Float64(1e-9),
            :fun_num => Int(4),
            :reg => "exponential regularization",
            :lambda => Float64(0.1),
            :theta => Float64(5),
            :data => "movielens1m"
        )
        
        # Extract sparse matrix data
        row, col, val = findnz(data)
        
        # Normalize values
        val = val .- mean(val)
        val = val ./ std(val)
        
        # Split into train and test (70-30 split)
        Random.seed!(30)  # For reproducibility
        idx = randperm(length(val))
        train_idx = idx[1:floor(Int, length(val)*0.7)]
        test_idx = idx[ceil(Int, length(val)*0.3):end]
        
        m, n = size(data)
        
        # Create sparse train data
        train_data = sparse(row[train_idx], col[train_idx], val[train_idx], m, n)
        
        # Extract test data as individual arrays
        test_row = row[test_idx]
        test_col = col[test_idx]
        test_data = val[test_idx]
        
        # Initialize the factors
        R = randn(n, params[:maxR])
        
        # Use power_method with explicit integer argument
        U0 = power_method(train_data, R, 5, 1e-6)
        
        F = svd(U0' * train_data)
        V0 = F.Vt'
        
        # Set up parameters individually
        params[:test_row] = test_row
        params[:test_col] = test_col
        params[:test_data] = test_data
        params[:test_m] = m
        params[:test_n] = n
        params[:R] = R
        params[:U0] = U0
        params[:V0] = V0
        
        # Run algorithms
        println("\n=== Running DCA ===")
        result_dca = DCA(train_data, params[:lambda], params[:theta], params)
        
        println("\n=== Running iDCA ===")
        result_idca = iDCA(train_data, params[:lambda], params[:theta], params)
        
        println("\n=== Running DCAe ===")
        result_dcae = DCAe(train_data, params[:lambda], params[:theta], params)
        
        # Combine results
        results = Dict("DCA" => result_dca, "iDCA" => result_idca, "DCAe" => result_dcae)
        
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
        println("\nMovieLens dataset not found at $data_path. Skipping this example.")
        println("To run this example, make sure the movielens1m.mat file is in the data directory.")
    end
end

# Run the examples
function main()
    println("DifferenceOfConvex Algorithms - Example Runner")
    println("=============================================")
    
    # Run synthetic example
    run_synthetic_example()
    
    # Run MovieLens example
    run_movielens_example()
end

# Execute main function if this script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
else
    main()  # Also run main if included as a module
end