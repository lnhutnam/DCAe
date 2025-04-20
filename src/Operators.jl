# Operators.jl - Proximal operators and update functions

using Polynomials  # Make sure this is imported for Poly

"""
    make_update_iDCAe(grad_U, grad_V, grad_h_U, grad_h_V, c_1, c_2, L, lam, ga, fun_num)

Calculate updates for U and V matrices based on proximal operators and gradients.

# Arguments
- `grad_U`: Gradient with respect to U
- `grad_V`: Gradient with respect to V
- `grad_h_U`: Gradient of the concave part with respect to U
- `grad_h_V`: Gradient of the concave part with respect to V
- `c_1`, `c_2`: Constants used in the gradient step
- `L`: Lipschitz constant
- `lam`: Regularization parameter
- `ga`: Regularization shape parameter
- `fun_num`: Function type indicator

# Returns
- Updated matrices U and V
"""
function make_update_iDCAe(grad_U, grad_V, grad_h_U, grad_h_V, c_1, c_2, L, lam, ga, fun_num)
    tpk = grad_U / L - grad_h_U
    tqk = grad_V / L - grad_h_V
    
    # Apply appropriate proximal operator based on function type
    if fun_num == 1
        # L1 norm proximal operator (soft thresholding)
        pk = max.(0, abs.(tpk) .- lam/L) .* sign.(-tpk)
        qk = max.(0, abs.(tqk) .- lam/L) .* sign.(-tqk)
    elseif fun_num == 4
        # Exponential regularization proximal operator
        pk = max.(0, abs.(tpk) .- ga*lam/L) .* sign.(-tpk)
        qk = max.(0, abs.(tqk) .- ga*lam/L) .* sign.(-tqk)
    elseif fun_num == 6
        # One-sided regularization
        pk = max.(0, -tpk .- lam*ga/L)
        qk = max.(0, -tqk .- lam*ga/L)
    else
        error("Unsupported function number: $fun_num")
    end
    
    # Solve cubic equation to find optimal scaling
    pk_norm_sq = sum(pk.^2)
    qk_norm_sq = sum(qk.^2)
    coeff = [c_1*(pk_norm_sq + qk_norm_sq), 0, c_2, -1]
    
    # Fixed cubic equation solution using Polynomials.jl
    p = Polynomial(reverse(coeff))
    r_vals = roots(p)
    
    # Filter for real positive roots
    real_positive = filter(r -> isreal(r) && real(r) > 0, r_vals)
    
    if isempty(real_positive)
        r = 1.0  # fallback value
    else
        r = real(real_positive[end])
    end
    
    U = r * pk
    V = r * qk
    
    return U, V
end

"""
    hard_threshold(U, lam)

Perform hard thresholding - keep only the lam largest elements.

# Arguments
- `U`: Input matrix
- `lam`: Number of elements to keep

# Returns
- Thresholded matrix
"""
function hard_threshold(U, lam)
    tpk = vec(U)
    sorted_indices = sortperm(abs.(tpk), rev=true)
    
    result = zeros(size(U))
    flat_result = vec(result)
    
    for i in 1:min(lam, length(tpk))
        flat_result[sorted_indices[i]] = tpk[sorted_indices[i]]
    end
    
    return reshape(flat_result, size(U))
end