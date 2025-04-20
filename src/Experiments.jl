# experiment.jl - Functions for running experiments with DCA algorithms

"""
    run_experiment(dataset_file, params = Dict())

Run a comparison experiment using all three DCA variants.

# Arguments
- `dataset_file`: Path to the dataset file (.jld)
- `params`: Optional dictionary of parameters

# Returns
- Dictionary containing results for each algorithm
"""
function run_experiment(dataset_file, params = Dict())
    # Default parameters
    default_params = Dict(
        :maxR => 5,
        :maxtime => 20,
        :maxIter => 20000,
        :tol => 1e-9,
        :fun_num => 4,
        :reg => "exponential regularization",
        :lambda => 0.1,
        :theta => 5
    )
    
    # Merge provided params with defaults
    for (k, v) in default_params
        if !haskey(params, k)
            params[k] = v
        end
    end
    
    # Load dataset
    println("Loading dataset from $dataset_file")
    data = load(dataset_file)
    
    # Extract data and preprocess
    if haskey(data, "data")
        matrix_data = data["data"]
    else
        matrix_data = data # Assuming the data is the matrix itself
    end
    
    println("Dataset loaded, size: $(size(matrix_data))")
    
    row, col, val = findnz(matrix_data)
    
    # Normalize values
    val = val .- mean(val)
    val = val ./ std(val)
    
    # Split into train and test
    Random.seed!(42)  # For reproducibility
    idx = randperm(length(val))
    train_idx = idx[1:floor(Int, length(val)*0.7)]
    test_idx = idx[ceil(Int, length(val)*0.3):end]
    
    m, n = size(matrix_data)
    
    # Create sparse train data
    train_data = sparse(row[train_idx], col[train_idx], val[train_idx], m, n)
    
    # Set up test parameters
    test_params = Dict(
        :row => row[test_idx],
        :col => col[test_idx],
        :data => val[test_idx],
        :m => m,
        :n => n
    )
    
    params[:test] = test_params
    params[:data] = basename(dataset_file)
    
    # Initialize U0 and V0
    R = randn(n, params[:maxR])
    params[:R] = R
    
    U0 = power_method(train_data, R, params[:maxR], 1e-6)
    
    F = svd(U0' * train_data)
    V0 = F.Vt'
    
    params[:U0] = U0
    params[:V0] = V0
    
    # Run algorithms
    println("\n=== Running DCA ===")
    out1 = DCA(train_data, params[:lambda], params[:theta], params)
    
    println("\n=== Running iDCA ===")
    out2 = iDCA(train_data, params[:lambda], params[:theta], params)
    
    println("\n=== Running DCAe ===")
    out3 = DCAe(train_data, params[:lambda], params[:theta], params)
    
    return Dict("DCA" => out1, "iDCA" => out2, "DCAe" => out3)
end

"""
    plot_results(results)

Generate plots comparing the performance of DCA variants.

# Arguments
- `results`: Dictionary of results from run_experiment()

# Returns
- Tuple of plots (objective_plot, rmse_plot)
"""
function plot_results(results)
    using Plots
    
    # Plot objective values
    p1 = plot(xlabel="CPU time (s)", ylabel="Objective value (log scale)", 
              title="Objective Value Comparison", legend=:topright)
    
    for (method, result) in results
        plot!(p1, result[:Time], log.(result[:obj]), label=method, 
              linewidth=2, marker=:auto, markersize=3)
    end
    
    # Plot RMSE
    p2 = plot(xlabel="CPU time (s)", ylabel="RMSE", 
              title="RMSE Comparison", legend=:topright)
    
    for (method, result) in results
        plot!(p2, result[:Time], result[:RMSE], label=method, 
              linewidth=2, marker=:auto, markersize=3)
    end
    
    # Plot sparsity
    p3 = plot(xlabel="Iteration", ylabel="Non-zero elements (%)", 
              title="Sparsity Comparison", legend=:topright)
    
    for (method, result) in results
        iterations = 0:length(result[:nnzUV][:,1])-1
        plot!(p3, iterations, 100 * result[:nnzUV][:,1], label="$(method) (U)", 
              linewidth=2, linestyle=:solid)
        plot!(p3, iterations, 100 * result[:nnzUV][:,2], label="$(method) (V)", 
              linewidth=2, linestyle=:dash)
    end
    
    return p1, p2, p3
end

"""
    analyze_results(results)

Generate a detailed analysis of the results.

# Arguments
- `results`: Dictionary of results from run_experiment()

# Returns
- Dictionary with analysis metrics
"""
function analyze_results(results)
    analysis = Dict()
    
    for (method, result) in results
        # Compute convergence rate (average obj decrease per iteration)
        n_iter = length(result[:obj]) - 1
        if n_iter > 0
            conv_rate = (result[:obj][1] - result[:obj][end]) / n_iter
        else
            conv_rate = 0.0
        end
        
        # Time to reach specific objective thresholds
        if !hasattr(analysis, :thresholds)
            # Find minimum objective across all methods
            min_obj = minimum([minimum(res[:obj]) for (_, res) in results])
            # Create thresholds at increasing percentages of the way from max to min
            max_obj = maximum([res[:obj][1] for (_, res) in results])
            thresholds = [max_obj - p * (max_obj - min_obj) for p in [0.25, 0.5, 0.75, 0.9]]
            analysis[:thresholds] = thresholds
        end
        
        # Find times to reach thresholds
        times_to_threshold = []
        for threshold in analysis[:thresholds]
            idx = findfirst(x -> x <= threshold, result[:obj])
            if idx !== nothing && idx > 1
                push!(times_to_threshold, result[:Time][idx])
            else
                push!(times_to_threshold, Inf)
            end
        end
        
        # Final sparsity
        final_sparsity_U = result[:nnzUV][end, 1]
        final_sparsity_V = result[:nnzUV][end, 2]
        
        # Store analysis
        analysis[method] = Dict(
            :final_obj => result[:obj][end],
            :final_rmse => result[:RMSE][end],
            :n_iterations => n_iter,
            :total_time => result[:Time][end],
            :avg_time_per_iter => result[:Time][end] / n_iter,
            :conv_rate => conv_rate,
            :times_to_threshold => times_to_threshold,
            :final_sparsity => (U=final_sparsity_U, V=final_sparsity_V)
        )
    end
    
    return analysis
end
