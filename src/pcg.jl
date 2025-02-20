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

#= Computes in place v = Pₐr where Pₐ is the orthogonal projector operator onto the null space of a matrix A
Presented version does so by the normal equations approach, which involves the Cholesky factorization of AAᵀ
=#

function projection!(v::Vector{T}, A::Matrix{T}, chol_A::Cholesky{T,Matrix{T}}, r::Vector{T}) where T
    n = size(A,2)
    w, y = Vector{T}(undef,m), Vector{T}(undef,m) # auxiliary vectors

    y[:] = chol_A.L \ (A*r)
    w[:] = chol_A.U \ y
    v[:] = r - A'*w
    return
end


#= Apply the projected conjugate gradient method with a starting point x
Vector x is modified in place troughout the iterations =#
function projected_cg!(x::Vector{T}, 
    H::Matrix{T}, c::Vector{T}, 
    A::Matrix{T}, chol_A::Cholesky{T,Matrix{T}},
    ϵ::T, max_iter::Int; 
    verbose::Bool=false) where T

(_,n) = size(A)
r,v,p, Hp = Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n)

# Initialization

r[:] = H*x - c
projection!(v, A, chol_A, r)
rtv = vdot(r,v)
p[:] = -v[:]
terminated = abs(rtv) < ϵ
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
    terminated = iter > max_iter || abs(rtv) < ϵ
    verbose && @show iter
end
return
end

#= Solve minₓ xᵀHx/2 - cᵀx s.t. Ax = b
with H ≻ 0
by the projected conjugate gradient method (unpreconditionned version).
Returns the solution x
=#

# Method with a given starting point
function solve_quadratic(H::Matrix{T}, c::Vector{T}, A::Matrix{T}, b::Vector{T}, x0::Vector{T};
    ϵ::T = T(1e-6), max_iter::Int=100, verbose::Bool=false) where T

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
    ϵ::T = T(1e-6), max_iter::Int=100, verbose::Bool=false) where T

    (_,n) = size(A)
    x = Vector{T}(undef,n)
    chol_A = cholesky(A*A')
    solve_normal_equations!(A, chol_A, b, x)
    projected_cg!(x,H,c,A,chol_A,ϵ,max_iter)

    return x
end
