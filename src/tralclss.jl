#= TRALCLSS stands for Trust Region Augmented Lagrangian Constrainted Least Squares Solver =#


function tralclss()
    return
end


function minor_iterate(x::Vector{T},
                       w::Vector{T},
                       H::Matrix{T},
                       A::Matrix{T},
                       chol_AAᵀ::Cholesky{T,Matrix{T}},
                       fix_bounds::Vector{Int},
                       ℓ::Vector{T},
                       u::Vector{T},
                       Δ::T,
                       ε::T,
                       max_iter::Int,
                       verbose::Bool=false) where T
    @assert size(w,1) == size(x,1) "Current iterate and step arrays do not have same size"
    w[:] = x[:]

    # Bounds of the problem
    ℓ_bar = map(t -> max(t,-Δ), ℓ - x)
    u_bar = map(t -> min(t,Δ), u - x)
    chol_aug_aat = cholesky_aug_aat(A, fix_bounds, chol_AAᵀ)
    projected_cg!(w,H,c,A,chol_AAᵀ, chol_aug_aat, fix_bounds, ℓ_bar, u_bar, Δ, ε, max_iter)

    axpy!(-1,x,w) # minor step
    β = projected_search(w)

    # Update minor iterate
    axpy!(β,w,x)
    return
end

#= Projected search along the minor step w=#
function projected_search(w::Vector{T}) where T
    β = one(T)
    return β
end

#= Compute Cauchy point, that is also the first minor iterate =#

function cauchy_point!(x::Vector)
    return
end
