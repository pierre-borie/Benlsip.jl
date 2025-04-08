#= TRALCLSS stands for Trust Region Augmented Lagrangian Constrainted Least Squares Solver =#


function tralclss(x0::Vector{T},
                  λ0::Vector{T},
                  μ::T) where T
    return
end


#= Solves the sub problem of an outer iteration
Approximately minimize the Augmented Lagrangian function with respect to the primal variable with tolerance ω > 0=#
function inner_iteration(x0::Vector{T},
                         ) where T
    return
end


#= Compute a minor iterate
First compute a descent direction by approximately applying the conjugate gradient method =# 
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
    pcg_status = projected_cg!(w,H,c,A,chol_AAᵀ, chol_aug_aat, fix_bounds, ℓ_bar, u_bar, Δ, ε, max_iter)

    axpy!(-1,x,w) # minor step
    α = projected_search(w)

    # Update minor iterate
    axpy!(α,w,x)
    return pcg_status
end

#= Projected search along the minor step w=#
function projected_search(w::Vector{T}) where T
    β = one(T)
    return β
end

#= Compute an approximate Cauchy point by finding the first local minimum of a piecewise quadratic path =#

function cauchy_point!(x::Vector{T},
                       s_gc::Vector{T},
                       ∇f::Vector{T},
                       H::Matrix{T},
                       A::Matrix{T},
                       chol_AAᵀ::Cholesky{T,Matrix{T}},
                       ℓ::Vector{T},
                       u::Vector{T},
                       Δ::T,
                       verbose::Bool=false) where T
    (m,n) = size(A)
    zero_T = zero(T)
    # Buffer initialization
    s_i, e_i = zeros(T,n), zeros(T,n)
    d, d_i, Hdi = Vector{T}(undef,n), Vector{T}(undef,n), Vector{T}(undef,n)
    
    # Orthogonal projection of ∇f onto the null space of A
    projection!(d,A,chol_AAᵀ,∇f)
    d_i[:] = d[:]
    ℓ_bar = map(t -> max(t,-Δ), ℓ - x)
    u_bar = map(t -> min(t,Δ), u - x)

    # Breakpoints
    t_b, t_b_sorted = break_points(d, ℓ_bar, u_bar)

    t_i = zero_T
    for j in findall(isapprox(t_i),t_b)
        e_i[j] = d[j]
    end
    axpy!(-1,e_i,d_i)

    gtdi = dot(∇f,d_i)
    mul!(Hdi,H,d_i)

    minima_found = false
    while !minima_found || !isempty(t_b_sorted)

        # Slope and curvature
        qi_p = gtdi + dot(s_i,Hdi)
        qi_pp = dot(d_i,Hdi)
        Δt = (isapprox(zero_T)(qi_pp) ? zero_T : -qi_p / qi_pp)
        t_ip1 = popfirst!(t_b_sorted)

        if qi_p ≥ 0
            s_gc[:] = s_i[:]
            minima_found = true
        elseif qi_pp > 0 && Δt > t_ip1 - t_i
            s_gc[:] = s_i + Δt*d_i[:]
            minima_found = true
        else # Prepare for the next interval
            j_break = findall(isapprox(t_ip1),t_b)
            Δt = t_ip1 - t_ip
            axpy!(Δt,d_i,s_i)
            e_i[:] = [(j in j_break ? d[j] : zero_T) for j=1:n]
            axpy(-1,e_i,d_i)
            # Update gᵀd and Hd
            gtdi -= dot(∇f,e_i)
            mul!(Hdi,H,e_i,-1,1)
        end
    end
    
    return
end

function break_points(d::Vector{T},l::Vector{T},u::Vector{T}) where T
    @assert all(<(0),l) && all(>(0),u) "Upper and lower bounds must be of opposite sign"
    
    t_b = zeros(T, size(d,1))
    for i ∈ axes(d,1)
        if d[i] < 0
            t_b[i] = l/d[i]
        elseif d[i] > 0
            t_b[i] = u/d[i]
        end
    end
    # Ordered values
    t_b_sorted = sort(unique(t_b))
    filter(!isapprox(zero(T),t_b_sorted))  # Get rid of zero value breakpoint for convenience
    
    return t_b, t_b_sorted
end


