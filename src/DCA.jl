# DCA.jl - Basic Difference of Convex Algorithm implementation

"""
    DCA(D, lam, theta, para)

Implement the basic Difference of Convex Algorithm for matrix factorization.

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
function DCA(D, lam, theta, para)
    output = Dict{Symbol, Any}()
    output[:method] = "DCA"
    
    max_R = get(para, :maxR, min(size(D)...))
    max_iter = para[:maxIter]
    tol = para[:tol]
    
    row, col, data = findnz(D)
    m, n = size(D)
    
    R = para[:R]
    U0 = para[:U0]
    U1 = copy(U0)
    
    V0 = para[:V0]'
    V1 = copy(V0)
    
    spa = sparse(row, col, data, m, n)
    
    c_1 = 3
    c_2 = norm(data)
    
    # Initialize tracking variables - use max_iter + 2 to ensure enough space
    obj = zeros(max_iter + 2)
    obj_y = zeros(max_iter + 2)
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
    
    obj[1] = obj_func(part0, U0, V0, lam, fun_num, ga)
    obj_y[1] = obj[1]
    
    L = 1
    Lls[1] = L
    
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
        println("method: $(output[:method]) data: $(para[:data]) RMSE $(RMSE[1]) obj $(obj[1])")
    end
    
    # Initialize iteration counter outside the loop to ensure it's accessible after the loop ends
    i = 0
    
    # Force at least 3 iterations to get a proper line
    min_iterations = 3
    
    # Main algorithm loop
    for iter = 1:max_iter
        i = iter  # Update the outer i to track the current iteration
        start_time = time()
        
        y_U = copy(U1)
        y_V = copy(V1)
        
        y_obj = obj[i]
        no_acceleration[i] = i
        
        obj_y[i] = y_obj
        
        U0 = copy(U1)
        V0 = copy(V1)
        
        set_sparse_values!(spa, part0, length(part0))
        
        # Compute gradients
        grad_U = -spa * y_V'
        grad_V = -y_U' * spa
        
        if fun_num == 4
            grad_U = grad_U - lam * ga * (1 .- exp.(-ga .* abs.(y_U))) .* sign.(y_U)
            grad_V = grad_V - lam * ga * (1 .- exp.(-ga .* abs.(y_V))) .* sign.(y_V)
        end
        
        norm_y = norm(y_U)^2 + norm(y_V)^2
        grad_h1_U = y_U * norm_y
        grad_h1_V = y_V * norm_y
        
        grad_h2_U = y_U
        grad_h2_V = y_V
        
        grad_h_U = c_1 * grad_h1_U + c_2 * grad_h2_U
        grad_h_V = c_1 * grad_h1_V + c_2 * grad_h2_V
        
        obj_h_y = c_1 * 0.25 * (norm_y^2) + c_2 * 0.5 * norm_y
        
        # Initialize x_obj before the loop
        x_obj = 0.0
        
        # Update U and V
        for inneriter in 1:1
            U1, V1 = make_update_iDCAe(grad_U, grad_V, grad_h_U, grad_h_V, c_1, c_2, L, lam, ga, 6)
            
            norm_x = norm(U1)^2 + norm(V1)^2
            
            delta_U = U1 - y_U
            delta_V = V1 - y_V
            
            obj_h_x = c_1 * 0.25 * (norm_x^2) + c_2 * 0.5 * norm_x
            
            reg = obj_h_x - obj_h_y - sum(delta_U .* grad_h_U) - sum(delta_V .* grad_h_V)
            
            part0 = sparse_inp(U1', V1, row, col)
            part0 = data - part0
            
            x_obj = obj_func(part0, U1, V1, lam, fun_num, ga)
        end
        
        Lls[i+1] = L
        Ils[i+1] = 1
        
        if i > 1
            delta = (obj[i] - x_obj) / x_obj
        else
            delta = Inf
        end
        
        Time[i+1] = time() - start_time
        obj[i+1] = x_obj
        
        println("iter: $i; obj: $x_obj (dif: $delta); rank $(para[:maxR]); lambda: $lam; L $L; time: $(Time[i+1]); " *
                "nnz U: $(count(!iszero, U1)/(size(U1,1)*size(U1,2))); nnz V $(count(!iszero, V1)/(size(V1,1)*size(V1,2)))")
        
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
    println("DCA completed with $final_iter data points")
    
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