##### Projection related methods

#= The goal is to compute 'v = Pₐr' where 'Pₐ' is the orthogonal projection operator onto the set 
'{x | Ax = 0, xᵢ = 0, i ∈ fix_bounds}', rewritten '{x | Ãx = 0}'

* 'A' is a 'm × n' matrix

* 'fix_bounds' is encoded by a 'BitVector' such that the indices of the 'true' elements form a subset of '{1,...n}'

* 'Ã' is an (m+p) × n augmented matrix representing the feasible set  

Uses the normal equations approach, which involves the Cholesky decomposition of 'ÃÃᵀ' =# 


#= Forms the Cholesky decomposition of ÃÃᵀ 
Computation exploits its block structure =#

function cholesky_aug_aat(
    A::Matrix{T}, 
    fix_bounds::BitVector, 
    chol_aat::Cholesky{T,Matrix{T}}) where T

    (m,n) = size(A)
    p = count(fix_bounds)
    mpp = m+p
    @assert mpp < n 

    # Auxiliary buffer arrays 
    H = Matrix{T}(I,p,p)
    L = LowerTriangular(Matrix{T}(undef, mpp, mpp))
    
    A_act_cols = view(A,:,fix_bounds)
    G = chol_aat.L \ A_act_cols
    # TODO: implement a more efficient computation of I - GᵀG
    mul!(H, G', G, -1, 1) # forms I - GᵀG
    
    # Forms the L factor of ÃÃᵀ Cholesy decomposition
    L[1:m,1:m] = chol_aat.L
    L[m+1:end,1:m] = G'
    L[m+1:end,p+1:end] = cholesky(H).L  
    return Cholesky(L)
end

cholesky_aat(A::Matrix{T}) where T = Cholesky(A*A')

#= The following methods perform the left multiplication by 'Ã' and 'Ãᵀ' 
Methods are implemented in both in place and out-of-place versions =#

function mul_a_tilde!(
    y::Vector{T}, 
    A::Matrix{T}, 
    fix_bounds::BitVector, 
    x::Vector{T}) where T

    (m,_) = size(A)
    @assert m+count(fix_bounds) == size(y,1)

    mul!(view(y,1:m), A, x)
    y[m+1:end] .= x[fix_bounds]
    return
end


function mul_tr_a_tilde!(
    y::Vector{T}, 
    A::Matrix{T}, 
    fix_bounds::BitVector, 
    x::Vector) where T

    (m,n) = size(A)
    @assert n == size(y,1)

    mul!(y,A',x[1:m]) # Aᵀ  component    
    y[fix_bounds] .+= x[m+1:end] # Zᵀ component
    return
end

function mul_a_tilde(
    A::Matrix{T}, 
    fix_bounds::BitVector, 
    x::Vector{T}) where T

    y = Vector{T}(undef, size(A,1)+count(fix_bounds))
    mul_a_tilde!(y, A, fix_bounds, x)
    return y
end

function mul_tr_a_tilde(
    A::Matrix{T}, 
    fix_bounds::BitVector, 
    x::Vector{T}) where T

    y = Vector{T}(undef, size(A,2))
    mul_tr_a_tilde!(y, A, fix_bounds, x)
    return y
end

# Projections methods

function projection!(v::Vector{T}, 
    A::Matrix{T}, 
    chol_aat::Cholesky{T,Matrix{T}}, 
    r::Vector{T}) where T

    m = size(A,1)
    w, y = Vector{T}(undef,m), Vector{T}(undef,m) # auxiliary vectors

    y[:] = chol_aat.L \ (A*r)
    w[:] = chol_aat.U \ y
    v[:] = r - A'*w
    return
end

function projection!(v::Vector{T}, 
    A::Matrix{T}, 
    chol_aug_aat::Cholesky{T,Matrix{T}}, 
    fix_bounds::BitVector, 
    r::Vector{T}) where T

    (m,n) = size(A)
    p = count(fix_bounds)
    mpp = m+p
    @assert mpp < n 

    w, y = Vector{T}(undef,mpp), Vector{T}(undef,mpp) # auxiliary vectors
    

    # Solves the normal equations to compute the orthogonal projection
    y[:] = chol_aug_aat.L \ mul_a_tilde(A, fix_bounds, r)
    w[:] = chol_aug_aat.U \ y 
    v[:] = r - mul_tr_a_tilde(A, fix_bounds, w)  
    return
end

"""
    projection(A, chol_aat, r)

Compute the projection of vector 'r' onto the null space of matrix 'A'.

More precisely, solves the normal equations associated to the problem 'minᵥ {||v-r|| | Av = 0}' using the Cholesky factorization of 'AAᵀ'.
"""
function projection(
    A::Matrix{T}, 
    chol_aat::Cholesky{T,Matrix{T}},
    r::Vector{T}) where T

    v = Vector{T}(undef,size(r,1))
    projection!(v,A,chol_aat,r)
    return v
end

"""
    projection(A, chol_aug_aat, fix_bounds, r)

Compute the projection of vector 'r' onto the null space of matrix 'A' with some components fixed to 0.

More precisely, solves the normal equations associated to the problem 'minᵥ {||v-r|| | Av = 0, vᵢ = 0 for i ∈ fix_bounds}' using the Cholesky factorization of an augmented matrix.
"""
function projection(
    A::Matrix{T}, 
    chol_aug_aat::Cholesky{T,Matrix{T}}, 
    fix_bounds::BitVector, 
    r::Vector{T}) where T

    v = Vector{T}(undef,size(r,1))
    projection!(v,A,chol_aug_aat,fix_bounds,r)
    return v 
end

"""
    function projection_polyhedron(x,A,b,l,u;solver)

Compute the projection of vector 'x' onto the polyhedron '{v | Av = b, l ≤ v ≤ u}' by solving the associated minimum-distance quadratic program.

The default solver is 'Ipopt'.
"""
function projection_polyhedron(
    x::Vector{T}, 
    A::Matrix{T}, 
    b::Vector{T}, 
    l::Vector{T}, 
    u::Vector{T}; 
    solver = Ipopt.Optimizer) where T

    n = size(x,1)
    model = Model(solver)
    set_silent(model)
    set_attribute(model, "hessian_constant", "yes") # Option to make Ipopt assume it is a QP

    @variable(model, l[i] <= v[i=1:n] <= u[i])
    @constraint(model, A*v .== b)
    @objective(model, Min, dot(v-x,v-x))
    optimize!(model)

    return value.(v)
end

##### Methods relative to active bounds manipulation

# Return the indices of active Constraints

active_index(fix_bounds::BitVector) = [i for (i,val) in enumerate(fix_bounds) if val]
