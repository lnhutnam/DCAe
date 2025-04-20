# Operations.jl - Matrix operations for DifferenceOfConvex

"""
    partXY(U, V, row, col, data_length)

Compute dot products between corresponding columns of U and V 
at positions specified by row and col indices.

# Arguments
- `U`: First factor matrix (transposed)
- `V`: Second factor matrix
- `row`: Row indices of observed entries
- `col`: Column indices of observed entries
- `data_length`: Number of observed entries

# Returns
- Vector of dot products
"""
function partXY(U, V, row, col, data_length)
    result = zeros(data_length)
    for i in 1:data_length
        result[i] = dot(U[:, row[i]], V[:, col[i]])
    end
    return result
end

"""
    sparse_inp(U, V, row, col)

Compute matrix elements at specified positions.
Similar to partXY but with different interface.

# Arguments
- `U`: First factor matrix (transposed)
- `V`: Second factor matrix
- `row`: Row indices of observed entries
- `col`: Column indices of observed entries

# Returns
- Vector of computed elements
"""
function sparse_inp(U, V, row, col)
    data_length = length(row)
    result = zeros(data_length)
    for i in 1:data_length
        result[i] = dot(U[:, row[i]], V[:, col[i]])
    end
    return result
end

"""
    set_sparse_values!(spa, values, length_values)

Update the values of a sparse matrix in-place.

# Arguments
- `spa`: Sparse matrix to update
- `values`: New values
- `length_values`: Length of values vector
"""
function set_sparse_values!(spa, values, length_values)
    for i in 1:length_values
        spa.nzval[i] = values[i]
    end
end
