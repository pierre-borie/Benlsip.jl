export solve_quadratic

#= Computes the least norm vector satisfying Ax=b, i.e. x = Aᵀ(AAᵀ)⁻¹b by the normal equations approach
Vector x is computed in place 
Makes use of the Cholesy decomposition of A
=#
function solve_normal_equations!(A::Matrix{T}, chol_A::Cholesky{T,Matrix{T}}, b::Vector{T}, x::Vector{T}) where T
    (m,n) = size(A)
    w, y = Vector{T}(undef,m), Vector{T}(undef,m) # auxiliary vectors   
    w[:] = chol_A.L \ b
    y[:] = chol_A.U \ w
    x[:] = A'*y
    return
end

#= This method computes v = P̃ₐr where P̃ₐ is the orthogonal projection operator onto the set {x | 'Ax = 0'  and 'xᵢ = 0' for i ∈ 𝒜} =  {x | 'Bx = 0'} 

* 'A' is a m × n matrix

* 𝒜 ⊂ {1,...n} such that |𝒜| = p

Uses the normal equations approach, which involves the Cholesky decomposition of 'B'.

=#

# The following two functions efficiently compute (hopefully), respectively, the matrix vector products Ãx and Ãᵀx and stores the result in y.
function mul_A_tilde!(y::Vector{T}, A::Matrix{T}, fix_bounds::Vector{Int}, x::Vector{T}) where T

    (m,_) = size(A)
    @assert m+size(fix_bounds,1) == size(y,1)

    mul!(view(y,1:m), A, x)
    y[m+1:end] = x[fix_bounds]
    return
end


function mul_A_tildeT!(y::Vector{T}, A::Matrix{T}, fix_bounds::Vector{Int}, x::Vector) where T
    (m,n) = size(A)
    @assert n == size(y,1)

    mul!(y,A',x[1:m]) # Aᵀ  component

    # Zᵀ component
    for (k,i) ∈ enumerate(fix_bounds)
        y[i] += x[m+k]
    end
    return
end

# The following two functions efficiently compute (hopefully), respectively, the matrix vector products Ãx and Ãᵀx and return the result.
function mul_A_tilde(A::Matrix{T}, fix_bounds::Vector{Int}, x::Vector{T}) where T
    y = Vector{T}(undef, size(A,1)+size(fix_bounds,1))
    mul_A_tilde!(y, A, fix_bounds, x)
    return y
end

function mul_A_tildeT(A::Matrix{T}, fix_bounds::Vector{Int}, x::Vector{T}) where T
    y = Vector{T}(undef, size(A,2))
    mul_A_tildeT!(y, A, fix_bounds, x)
    return y
end

# Forms the cholesky decomposition of ÃÃᵀ

function cholesky_aug_aat(A::Matrix{T}, fix_bounds::Vector{Int}, chol_AAᵀ::Cholesky{T,Matrix{T}}) where T
    (m,n) = size(A)
    p = size(fix_bounds,1)
    mpp = m+p
    @assert mpp < n 

    # Auxiliary buffer arrays 
    H = Matrix{T}(I,p,p)
    L = LowerTriangular(Matrix{T}(undef, mpp, mpp))
    
    A_act_cols = view(A,:,fix_bounds)
    G = chol_AAᵀ.L \ A_act_cols
    mul!(H, G', G, -1, 1) # forms I - GᵀG
    
    # Forms the L factor of BBᵀ Cholesy decomposition
    L[1:m,1:m] = chol_AAᵀ.L
    L[m+1:end,1:m] = G'
    L[m+1:end,p+1:end] = cholesky(H).L  
    return Cholesky(L)
end

##### Projections

#= Function projection! computes the orthogonal projection of a vector r onto a polyhedral set and stores the result in vector v
Presented version does so by the normal equations approach, which involves the Cholesky factorization of AAᵀ

For the first method, the projection lies in the null space of a matrix A

=#

function projection!(v::Vector{T}, A::Matrix{T}, chol_AAᵀ::Cholesky{T,Matrix{T}}, r::Vector{T}) where T
    m = size(A,1)
    w, y = Vector{T}(undef,m), Vector{T}(undef,m) # auxiliary vectors

    y[:] = chol_AAᵀ.L \ (A*r)
    w[:] = chol_AAᵀ.U \ y
    v[:] = r - A'*w
    return
end
 

function projection!(v::Vector{T}, A::Matrix{T}, chol_BBᵀ::Cholesky{T,Matrix{T}}, fix_bounds::Vector{Int}, r::Vector{T}) where T
    (m,n) = size(A)
    p = size(fix_bounds,1)
    mpp = m+p
    @assert mpp < n 

    w, y = Vector{T}(undef,mpp), Vector{T}(undef,mpp) # auxiliary vectors
    

    # Solves the normal equations to compute the orthogonal projection
    y[:] = chol_BBᵀ.L \ mul_A_tilde(A, fix_bounds, r)
    w[:] = chol_BBᵀ.U \ y 
    v[:] = r - mul_A_tildeT(A, fix_bounds, w)  
    return
end

function factor_to_boundary(p::Vector{T}, ℓ_bar::Vector{T}, u_bar::Vector{T}) where T
    i_negp = [i for i ∈ axes(p,1) if p[i] < 0]
    γ1 = minimum(ℓ_bar[i] for i ∈ i_negp)
    γ2 = minimum(u_bar[i] for i ∈ setdiff(axes(p,1), i_negp))
    return min(γ1, γ2)
end

##### Conjugate gradient 

#= Apply the projected conjugate gradient method with a starting point x
Vector x is modified in place troughout the iterations =#

function projected_cg!(x::Vector{T}, 
    H::Matrix{T}, c::Vector{T}, 
    A::Matrix{T}, chol_A::Cholesky{T,Matrix{T}},
    ε::T, max_iter::Int; 
    verbose::Bool=false) where T

    (_,n) = size(A)
    r,v,p, Hp = Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n)
    
    # Initialization

    r[:] = H*x - c
    projection!(v, A, chol_A, r)
    rtv = vdot(r,v)
    p[:] = -v[:]
    terminated = abs(rtv) < ε
    iter = 1
    while !terminated
        mul!(Hp, H, p) 
        α = rtv / dot(p,Hp)
        axpy!(α,p,x)
        axpy!(α,Hp,r)
        projection!(v, A, chol_A, r)
        verbose && @show A*v
        rtv_next = dot(r,v)
        β = rtv_next / rtv
        axpby!(-one(T), v, β, p)
        rtv = rtv_next
        iter += 1
        terminated = iter > max_iter || abs(rtv) < ε
        verbose && @show iter
    end
    return
end

function projected_cg!(x::Vector{T}, 
                       H::Matrix{T},
                       c::Vector{T}, 
                       A::Matrix{T},
                       chol_AAᵀ::Cholesky{T,Matrix{T}},
                       chol_aug_aat::Cholesky{T,Matrix{T}},
                       fix_bounds::Vector{Int},
                       ℓ_bar::Vector{T},
                       u_bar::Vector{T},
                       Δ::T,
                       ε::T,
                       max_iter::Int; 
                       verbose::Bool=false) where T

    (_,n) = size(A)
    r,v,p, Hp = Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n)
    
    # Initialization

    r[:] = H*x - c
    projection!(v, A, chol_aug_aat, fix_bounds, r)
    rtv = vdot(r,v)
    p[:] = -v[:]

    bound_hit = false
    stationary = abs(rtv) < ε
    iter = 1

    while !(stationary || bound_hit || max_iter_reached)
        mul!(Hp, H, p) 
        α = rtv / dot(p,Hp)

        # Distance to boundary
        
        γ = factor_to_boundary(p, ℓ_bar, u_bar)
        bound_hit = α > γ

        if bound_hit
            axpy!(γ,p,x)
        else 
            axpy!(α,p,x)
            axpy!(α,Hp,r)
            projection!(v, A, chol_aug_aat, fix_bounds, r)
            rtv_next = dot(r,v)
            β = rtv_next / rtv
            axpby!(-one(T), v, β, p)
            rtv = rtv_next
        end
        iter += 1

        stationary = abs(rtv) < ε
        max_iter_reached = iter > max_iter
        verbose && @show iter
    end

    status = if stationary
        :stationary
    elseif bound_hit
        :bound_hit
    elseif max_iter_reached
        :max_iter_reached
    end
    return status
end
#= Solve minₓ xᵀHx/2 - cᵀx s.t. Ax = b
with H ≻ 0
by the projected conjugate gradient method (unpreconditionned version).
Returns the solution x
=#

# Method with a given starting point
function solve_quadratic(H::Matrix{T}, c::Vector{T}, A::Matrix{T}, b::Vector{T}, x0::Vector{T};
    ε::T = T(1e-6), max_iter::Int=100, verbose::Bool=false) where T

    (m,n) = size(A)
    x = Vector{T}(undef,n)
    x[:] = x0[:]
    chol_A = cholesky(A*A')
    projected_cg!(x,H,c,A, chol_A, ε, max_iter)

    return x
end

#= Method without user starting point.  
Computes one by the normal equations approach =#

function solve_quadratic(H::Matrix{T}, c::Vector{T}, A::Matrix{T}, b::Vector{T};
    ε::T = T(1e-6), max_iter::Int=100, verbose::Bool=false) where T

    (_,n) = size(A)
    x = Vector{T}(undef,n)
    chol_A = cholesky(A*A')
    solve_normal_equations!(A, chol_A, b, x)
    projected_cg!(x,H,c,A,chol_A,ε,max_iter)

    return x
end
