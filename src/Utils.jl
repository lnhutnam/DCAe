# utils.jl - Utility functions for DifferenceOfConvex

"""
    obj_func(part, U, V, lam, fun_num, ga)

Calculate objective function value for the given parameters.

# Arguments
- `part`: Residual vector (data - prediction)
- `U`: First factor matrix
- `V`: Second factor matrix
- `lam`: Regularization parameter
- `fun_num`: Regularization function type (0=L0, 1=L1, 4=Exponential, 5=Cardinality)
- `ga`: Parameter for exponential regularization

# Returns
- Objective function value
"""
function obj_func(part, U, V, lam, fun_num, ga)
    if fun_num == 0
        return 0.5 * sum(part.^2) + lam * (count(!iszero, U) + count(!iszero, V))
    elseif fun_num == 1
        return 0.5 * sum(part.^2) + lam * (sum(abs.(U)) + sum(abs.(V)))
    elseif fun_num == 4
        return 0.5 * sum(part.^2) + lam * (sum(1 .- exp.(-ga .* abs.(U))) + sum(1 .- exp.(-ga .* abs.(V))))
    elseif fun_num == 5
        if count(!iszero, U) <= lam && count(!iszero, V) <= lam
            return 0.5 * sum(part.^2) + lam * (count(!iszero, U) + count(!iszero, V))
        else
            return Inf
        end
    else
        error("Unsupported regularization function number: $fun_num")
    end
end

"""
    MatCompRMSE(U, V, S, test_row, test_col, test_data)

Calculate RMSE on test data for matrix completion.

# Arguments
- `U`: First factor matrix
- `V`: Second factor matrix  
- `S`: Identity matrix or scaling matrix
- `test_row`: Row indices of test entries
- `test_col`: Column indices of test entries
- `test_data`: Values of test entries

# Returns
- RMSE value
"""
function MatCompRMSE(U, V, S, test_row, test_col, test_data)
    n_test = length(test_data)
    error_sum = 0.0
    
    for i in 1:n_test
        pred = 0.0
        for k in 1:size(U, 2)
            for j in 1:size(V, 2)
                pred += U[test_row[i], k] * S[k, j] * V[test_col[i], j]
            end
        end
        error_sum += (pred - test_data[i])^2
    end
    
    return sqrt(error_sum / n_test)
end

"""
    power_method(D, R, max_iter, tol)

Implement power method to find principal components.

# Arguments
- `D`: Data matrix (sparse)
- `R`: Random initialization matrix
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance

# Returns
- Matrix U with orthogonal columns
"""
function power_method(D, R, max_iter::Int, tol)
    m, n = size(D)
    r = size(R, 2)
    U = randn(m, r)
    
    for i in 1:max_iter
        old_U = copy(U)
        U = D * R
        
        for j in 1:r
            U[:, j] /= norm(U[:, j])
        end
        
        if norm(U - old_U) < tol
            break
        end
    end
    
    return U
end