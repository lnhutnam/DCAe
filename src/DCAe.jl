# DCAe.jl - DCA with Extrapolation implementation

"""
    DCAe(D, lam, theta, para)

Implement the DCA with Extrapolation algorithm for matrix factorization.
This version uses Nesterov-style acceleration.

# Arguments
- `D`: Sparse input data matrix
- `lam`: Regularization parameter
- `theta`: Parameter for exponential regularization (ga)
- `para`: Dictionary of additional parameters

# Returns
- Dictionary of results including:
  - U, V: Factorized matrices
  - obj: Objective function values
  - RMSE: Root Mean Square Error on test data
  - Time: Cumulative runtime
  - And other metrics
"""
function DCAe(D, lam, theta, para)
    output = Dict{Symbol, Any}()
    output[:method] = "DCAe"
    
    max_R = get(para, :maxR, min(size(D)...))
    max_iter = para[:maxIter]
    tol = para[:tol]
    
    row, col, data = findnz(D)
    m, n = size(D)
    
    R = para[:R]
    U0 = para[:U0]
    U1 = copy(U0)
    
    F = svd(U0' * D)
    V0 = F.Vt
    V1 = copy(V0)
    
    spa = sparse(row, col, data, m, n)
    
    c_1 = 3
    c_2 = norm(data)
    
    # Initialize tracking variables - use max_iter + 2 to ensure enough space
    obj = zeros(max_iter + 2)
    RMSE = zeros(max_iter + 2)
    train_RMSE = zeros(max_iter + 2)
    Time = zeros(max_iter + 2)
    Lls = zeros(max_iter + 2)
    Ils = zeros(max_iter + 2)
    nnz_UV = zeros(max_iter + 2, 2)
    no_acceleration = zeros(Int, max_iter + 2)
    
    part0 = partXY(U0', V0, row, col, length(data))
    part0 = data - part0
    
    ga = theta
    fun_num = para[:fun_num]
    
    L = 1
    uL = 1
    C = 0.9999
    
    norm1 = norm(U1)^2 + norm(V1)^2
    obj_1 = c_1 * 0.25 * (norm1^2) + c_2 * 0.5 * norm1
    obj_0 = obj_1
    
    x_obj = 0.5 * sum(part0.^2)
    obj[1] = x_obj + lam * (sum(1 .- exp.(-ga .* abs.(U0))) + sum(1 .- exp.(-ga .* abs.(V0))))
    
    c = 1
    
    # Test performance if requested - modified to use individual test parameters
    if haskey(para, :test_row) && haskey(para, :test_col) && haskey(para, :test_data)
        temp_S = Matrix(I, size(U1, 2), size(V1', 2))
        if para[:test_m] != m
            RMSE[1] = MatCompRMSE(V1', U1, temp_S, para[:test_row], para[:test_col], para[:test_data])
            train_RMSE[1] = sqrt(sum(part0.^2) / length(data))
        else
            RMSE[1] = MatCompRMSE(U1, V1', temp_S, para[:test_row], para[:test_col], para[:test_data])
            train_RMSE[1] = sqrt(sum(part0.^2) / length(data))
        end
        println("method: $(output[:method]) data: $(para[:data]) RMSE $(RMSE[1])")
    end
    
    # Initialize variables for proper scoping
    i = 0               # Initialize iteration counter
    y_U = copy(U1)      # Initialize y_U
    y_V = copy(V1)      # Initialize y_V
    norm_y = 0.0        # Initialize norm_y
    grad_h_yU = zeros(size(U1)) # Initialize gradient variables
    grad_h_yV = zeros(size(V1))
    obj_y = 0.0         # Initialize obj_y
    ibi = 0             # Initialize ibi counter
    
    # Initialize grad_U and grad_V to avoid undefined errors
    grad_U = zeros(size(U1))
    grad_V = zeros(size(V1))
    
    # Force at least 3 iterations to get a proper line
    min_iterations = 3
    
    # Main algorithm loop
    for iter = 1:max_iter
        i = iter  # Update the outer i to track the current iteration
        start_time = time()
        
        # Update extrapolation coefficient using Nesterov's formula
        c = (1 + sqrt(1 + 4 * c^2)) / 2
        bi = (c - 1) / c
        
        grad_h1_U = U1 * norm1
        grad_h1_V = V1 * norm1
        
        grad_h_U = c_1 * grad_h1_U + c_2 * U1
        grad_h_V = c_1 * grad_h1_V + c_2 * V1
        
        delta_U = U1 - U0
        delta_V = V1 - V0
        
        D_x = obj_0 - obj_1 - sum(delta_U .* grad_h_U) - sum(delta_V .* grad_h_V)
        
        # Extrapolation step
        for il in 1:1
            kappa = C * uL / (L + uL)
            
            for ibi = 1:30
                # Compute extrapolated point
                y_U = (1 + bi) * U1 - bi * U0
                y_V = (1 + bi) * V1 - bi * V0
                
                norm_y = norm(y_U)^2 + norm(y_V)^2
                grad_h1_yU = y_U * norm_y
                grad_h1_yV = y_V * norm_y
                
                grad_h2_yU = y_U
                grad_h2_yV = y_V
                
                grad_h_yU = c_1 * grad_h1_yU + c_2 * grad_h2_yU
                grad_h_yV = c_1 * grad_h1_yV + c_2 * grad_h2_yV
                
                obj_y = c_1 * 0.25 * (norm_y^2) + c_2 * 0.5 * norm_y
                
                D_y = obj_1 - obj_y - bi * sum(delta_U .* grad_h_yU) - bi * sum(delta_V .* grad_h_yV)
                
                # Check extrapolation condition
                if D_y <= kappa * D_x + 1e-10
                    break
                else
                    bi = 0.9 * bi  # Reduce extrapolation weight
                end
            end
            
            part1 = sparse_inp(y_U', y_V, row, col)
            part0 = data - part1
            
            set_sparse_values!(spa, part0, length(part0))
            
            # Calculate gradients here - this was causing the undefined variable error
            grad_U = -spa * y_V'
            grad_V = -y_U' * spa
        end
        
        U0 = copy(U1)
        V0 = copy(V1)
        
        # Initialize and calculate grad_U1 and grad_V1
        grad_U1 = copy(grad_U)
        grad_V1 = copy(grad_V)
        
        if fun_num == 4
            grad_U1 = grad_U - lam * ga * (1 .- exp.(-ga .* abs.(U1))) .* sign.(U1)
            grad_V1 = grad_V - lam * ga * (1 .- exp.(-ga .* abs.(V1))) .* sign.(V1)
        end
        
        obj_h_y = c_1 * 0.25 * (norm_y^2) + c_2 * 0.5 * norm_y
        obj_0 = obj_1
        
        # Initialize x_obj
        x_obj = 0.0
        
        # Update U and V
        for inneriter in 1:1
            U1, V1 = make_update_iDCAe(grad_U1, grad_V1, grad_h_yU, grad_h_yV, c_1, c_2, uL, lam, ga, 6)
            
            norm1 = norm(U1)^2 + norm(V1)^2
            obj_1 = c_1 * 0.25 * (norm1^2) + c_2 * 0.5 * norm1
            
            part0 = sparse_inp(U1', V1, row, col)
            part0 = data - part0
            
            x_obj = 0.5 * sum(part0.^2)
        end
        
        Lls[i+1] = L
        Ils[i+1] = 1
        
        obj_with_reg = x_obj + lam * (sum(1 .- exp.(-ga .* abs.(U1))) + sum(1 .- exp.(-ga .* abs.(V1))))
        
        if i > 1
            delta = (obj[i] - obj_with_reg) / x_obj
        else
            delta = Inf
        end
        
        Time[i+1] = time() - start_time
        obj[i+1] = obj_with_reg
        
        println("iter: $i; obj: $x_obj (dif: $delta); rank $(para[:maxR]); lambda: $lam; ibi: $ibi; uL: $uL; time: $(Time[i+1]); " *
                "nnz U: $(count(!iszero, U1)/(size(U1,1)*size(U1,2))); nnz V: $(count(!iszero, V1)/(size(V1,1)*size(V1,2)))")
        
        nnz_UV[i+1, 1] = count(!iszero, U1) / (size(U1, 1) * size(U1, 2))
        nnz_UV[i+1, 2] = count(!iszero, V1) / (size(V1, 1) * size(V1, 2))
        
        # Test performance if requested - modified to use individual test parameters
        if haskey(para, :test_row) && haskey(para, :test_col) && haskey(para, :test_data)
            temp_S = Matrix(I, size(U1, 2), size(V1', 2))
            if para[:test_m] != m
                RMSE[i+1] = MatCompRMSE(V1', U1, temp_S, para[:test_row], para[:test_col], para[:test_data])
                train_RMSE[i+1] = sqrt(sum(part0.^2) / length(data))
            else
                RMSE[i+1] = MatCompRMSE(U1, V1', temp_S, para[:test_row], para[:test_col], para[:test_data])
                train_RMSE[i+1] = sqrt(sum(part0.^2) / length(data))
            end
            println("method: $(output[:method]) data: $(para[:data]) RMSE $(RMSE[i+1])")
        end
        
        # Only check convergence after minimum iterations
        if i > min_iterations && abs(delta) < tol
            println("Convergence reached after $i iterations (delta = $delta < tol = $tol)")
            break
        end
        
        if sum(Time) > get(para, :maxtime, Inf)
            println("Maximum time limit reached after $i iterations")
            break
        end
    end
    
    final_iter = i + 1  # Because we added one more during the last loop iteration
    
    # Debug output
    println("DCAe completed with $final_iter data points")
    
    # Prepare output with the correct range
    output[:obj] = obj[1:final_iter]
    output[:Rank] = para[:maxR]
    output[:RMSE] = RMSE[1:final_iter]
    output[:trainRMSE] = train_RMSE[1:final_iter]
    
    Time = cumsum(Time)
    output[:Time] = Time[1:final_iter]
    output[:U] = U1
    output[:V] = V1
    output[:data] = para[:data]
    output[:L] = Lls[1:final_iter]
    output[:Ils] = Ils[1:final_iter]
    output[:nnzUV] = nnz_UV[1:final_iter, :]
    output[:no_acceleration] = no_acceleration[1:final_iter]
    output[:lambda] = lam
    output[:theta] = ga
    output[:reg] = get(para, :reg, "unknown")
    
    return output
end