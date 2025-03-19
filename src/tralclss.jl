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

function cauchy_point!(x::Vector{T},
                       ∇f::Vector{T},
                       H::Matrix{T},
                       A::Matrix{T},
                       chol_AAᵀ::Cholesky{T,Matrix{T}},
                       ℓ::Vector{T},
                       u::Vector{T},
                       Δ::T,
                       verbose::Bool=false) where T
    (m,n) = size(A)
    d = Vector{T}(undef,n)

    # Orthogonal projection of ∇f onto the null space of A
    projection!(v,A,chol_AAᵀ,∇f)
    ℓ_bar = map(t -> max(t,-Δ), ℓ - x)
    u_bar = map(t -> min(t,Δ), u - x)

    t_b = break_points(d, ℓ_bar, u_bar)
    return
end

function break_points(d::Vector{T},l::Vector{T},u::Vector{T})
    @assert all(<(0),l) && all(>(0),u) "Upper and lower bounds must be of opposite sign"
    
    t_b = zeros(size(d,1))
    for i ∈ axes(d,1)
        if d[i] < 0
            t_b[i] = l/d[i]
        elseif d[i] > 0
            t_b[i] = u/d[i]
        end
    end
    return t_b
end
