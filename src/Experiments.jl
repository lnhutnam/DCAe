# Experiments.jl - Functions for running experiments with DCA algorithms

using Plots

"""
    run_experiment(data_matrix, params = Dict())

Run a comparison experiment using all three DCA variants.

# Arguments
- `data_matrix`: Sparse data matrix
- `params`: Optional dictionary of parameters

# Returns
- Dictionary containing results for each algorithm
"""
function run_experiment(data_matrix, params = Dict())
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
    
    # Extract sparse matrix data
    row, col, val = findnz(data_matrix)
    
    # Normalize values if not already done
    if !haskey(params, :normalized) || !params[:normalized]
        val = val .- mean(val)
        val = val ./ std(val)
    end
    
    # Split into train and test if not provided
    m, n = size(data_matrix)
    
    if !haskey(params, :test_row)
        Random.seed!(42)  # For reproducibility
        idx = randperm(length(val))
        train_idx = idx[1:floor(Int, length(val)*0.7)]
        test_idx = idx[ceil(Int, length(val)*0.3):end]
        
        # Create sparse train data
        train_data = sparse(row[train_idx], col[train_idx], val[train_idx], m, n)
        
        # Set up test parameters
        params[:test_row] = row[test_idx]
        params[:test_col] = col[test_idx]
        params[:test_data] = val[test_idx]
        params[:test_m] = m
        params[:test_n] = n
    else
        train_data = data_matrix
    end
    
    if !haskey(params, :data)
        params[:data] = "dataset"
    end
    
    # Initialize U0 and V0 if not provided
    if !haskey(params, :R) || !haskey(params, :U0) || !haskey(params, :V0)
        R = randn(n, params[:maxR])
        params[:R] = R
        
        U0 = power_method(train_data, R, 5, 1e-6)
        
        F = svd(U0' * train_data)
        V0 = F.Vt'
        
        params[:U0] = U0
        params[:V0] = V0
    end
    
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
- Tuple of plots (objective_plot, rmse_plot, sparsity_plot)
"""
function plot_results(results)
    # Make sure we have results to plot
    if isempty(results)
        error("No results to plot")
    end
    
    # Extract data for plotting
    methods = keys(results)
    
    # Plot objective values - debug version with print statements
    println("Creating objective value plot...")
    p1 = plot(xlabel="CPU time (s)", ylabel="Objective value (log scale)", 
              title="Objective Value Comparison", legend=:topright)
    
    for method in methods
        result = results[method]
        println("Plotting $method with $(length(result[:Time])) time points and $(length(result[:obj])) objective values")
        
        # Make sure we have valid data
        if !isempty(result[:Time]) && !isempty(result[:obj]) && length(result[:Time]) == length(result[:obj])
            # Check for NaN or Inf values
            time_points = result[:Time]
            obj_values = log.(result[:obj])
            
            # Filter out any NaN or Inf values
            valid_indices = findall(t -> !isnan(t) && !isinf(t), obj_values)
            if !isempty(valid_indices)
                time_points = time_points[valid_indices]
                obj_values = obj_values[valid_indices]
                
                # Plot with explicit marker to ensure visibility
                plot!(p1, time_points, obj_values, 
                      label=method, linewidth=2, marker=:circle, markersize=3)
                
            end
        end
    end
    
    # Plot RMSE
    println("Creating RMSE plot...")
    p2 = plot(xlabel="CPU time (s)", ylabel="RMSE", 
              title="RMSE Comparison", legend=:topright)
    
    for method in methods
        result = results[method]
        if !isempty(result[:Time]) && !isempty(result[:RMSE]) && length(result[:Time]) == length(result[:RMSE])
            # Filter out any NaN or Inf values
            time_points = result[:Time]
            rmse_values = result[:RMSE]
            
            valid_indices = findall(t -> !isnan(t) && !isinf(t), rmse_values)
            if !isempty(valid_indices)
                time_points = time_points[valid_indices]
                rmse_values = rmse_values[valid_indices]
                
                plot!(p2, time_points, rmse_values, 
                      label=method, linewidth=2, marker=:circle, markersize=3)
            end
        end
    end
    
    # Plot sparsity
    println("Creating sparsity plot...")
    p3 = plot(xlabel="Iteration", ylabel="Non-zero elements (%)", 
              title="Sparsity Comparison", legend=:topright)
    
    for method in methods
        result = results[method]
        if !isempty(result[:nnzUV])
            # For sparsity plot, we use iterations rather than time
            iterations = 0:size(result[:nnzUV], 1)-1
            
            # Make sure we have something to plot
            if length(iterations) > 1
                # Plot U sparsity
                plot!(p3, iterations, 100 * result[:nnzUV][:, 1], 
                      label="$(method) (U)", linewidth=2, linestyle=:solid, marker=:circle, markersize=3)
                
                # Plot V sparsity
                plot!(p3, iterations, 100 * result[:nnzUV][:, 2], 
                      label="$(method) (V)", linewidth=2, linestyle=:dash, marker=:square, markersize=3)
            end
        end
    end
    
    println("Plotting complete.")
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
    
    # Find minimum objective across all methods
    min_obj = minimum([minimum(res[:obj]) for (_, res) in results])
    # Create thresholds at increasing percentages of the way from max to min
    max_obj = maximum([res[:obj][1] for (_, res) in results])
    thresholds = [max_obj - p * (max_obj - min_obj) for p in [0.25, 0.5, 0.75, 0.9]]
    analysis[:thresholds] = thresholds
    
    for (method, result) in results
        # Compute convergence rate (average obj decrease per iteration)
        n_iter = length(result[:obj]) - 1
        if n_iter > 0
            conv_rate = (result[:obj][1] - result[:obj][end]) / n_iter
        else
            conv_rate = 0.0
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