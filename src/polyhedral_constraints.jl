mutable struct MixedConstraints{T<:Real} <: TralcnllsData
    lineq::Matrix{T}
    xlow::Vector{T}
    xupp::Vector{T}
    fixvars::BitVector
    chol::Cholesky{T,Matrix{T}}
end

function MixedConstraints(
        A::Matrix{T},
        chol_aat::Cholesky{T,Matrix{T}}; 
        l::Vector{T} = fill(T(-Inf),size(A,2)), 
        u::Vector{T}=fill(T(Inf),size(A,2))) where {T<:Real}
    
    fixed = BitVector(undef,size(A,2))
    MixedConstraints(A,l,u,fixed,chol_aat)
end   

function MixedConstraints(
        A::Matrix{T},
        chol_aat::Cholesky{T,Matrix{T}},
        fixed::BitVector; 
        l::Vector{T} = fill(T(-Inf),size(A,2)), 
        u::Vector{T}=fill(T(Inf),size(A,2))) where {T<:Real}
    
    chol = (any(fixed) ? cholesky_aug_aat(A, fixed, chol_aat) : chol_aat)
    MixedConstraints(A,l,u,fixed,chol)
end  

#= Forms the Cholesky decomposition of ÃÃᵀ 
Computation exploits its block structure =#

function cholesky_aug_aat(
    A::Matrix{T}, 
    fix_bounds::BitVector, 
    chol_aat::Cholesky{T,Matrix{T}}) where T

    (m,n) = size(A)
    p = count(fix_bounds)
    mpp = m+p
    @assert mpp <= n 

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
    L[m+1:end,m+1:end] = cholesky(H).L  
    return Cholesky(L)
end

# Perform the Cholesky decomposition update on the representation 'lincons'
function update_chol!(
        lincons::MixedConstraints{T}, 
        chol_aat::Cholesky{T,Matrix{T}}) where T
    
    lincons.chol = cholesky_aug_aat(lincons.lineq, lincons.fixvars, chol_aat)
    return
end

#= The two following methods perform respectively the left multiplication by 'Ã' and 'Ãᵀ' =#

function left_mul_tr(lincons::MixedConstraints{T}, y::Vector{T}) where T
    
    (m,n) = size(lincons.lineq)
    x = Vector{T}(undef,n)
    
    if any(lincons.fixvars)
        mul!(x,lincons.lineq',y[1:m])
        x[lincons.fixvars] .+= y[m+1:end]
    else
        mul!(x,lincons.lineq',y)
    end
    return x
end

function left_mul(lincons::MixedConstraints{T}, x::Vector{T}) where T
    
    (m,_) = size(lincons.lineq)
    y = Vector{T}(undef, m+count(lincons.fixvars))

    if any(lincons.fixvars)
        mul!(view(y,1:m), lincons.lineq, x)
        y[m+1:end] .= x[lincons.fixvars]
    else
        mul!(y, lincons.lineq, x)
    end
    return y
end
    

#=
In place versions of the projection methods onto, respectively, the nullspace of 'A' and sets of the form '{v | Av = 0, vᵢ = 0 for i ∈ fixed variables}'
=#
function projection!(
    v::Vector{T},
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

function projection!(
    v::Vector{T}, 
    lincons::MixedConstraints{T}, 
    r::Vector{T}) where T

    (m,_) = size(lincons.lineq)
    mpp = m + count(lincons.fixvars)
    @assert mpp <= n 

    w, y = Vector{T}(undef,mpp), Vector{T}(undef,mpp) # auxiliary vectors
    
    y[:] = lincons.chol.L \ left_mul(lincons,r) 
    w[:] = lincons.chol.U \ y 
    v[:] = r - left_mul_tr(lincons,w)
    return
end


"""
    projection!(lincons,r)

Computes and returns the projection of vector 'r' onto the null space of a matrix 'Ã' represented by 'lincons' (see [`MixedConstraints`](@ref)).

The nullspace of 'Ã' corresponds to the feasible set '{v | Av = 0, vᵢ = 0 for i ∈ lincons.fixvars}'.

The projection is computed by solving the normal equations associated to the problem 'minᵥ {||v-r|| | Ãv = 0}' using the Cholesky factorization of 'ÃÃᵀ'.

If there are no fixed variables, i.e. 'Ã = A', then simply perfoms the projection onto the nullspace of 'A'.
"""
function projection(lincons::MixedConstraints{T}, r::Vector{T}) where T
        
    v = Vector{T}(undef,size(r,1))

    if any(lincons.fixvars)
        projection!(v,lincons,r)
    else
        projection!(v,lincons.lineq,lincons.chol,r)
    end
    return v
end