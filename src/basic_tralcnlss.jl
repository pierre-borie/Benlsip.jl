#= Structure and methods associated to the augmented Lagrangian

Contains
- evaluation of the objective 
- evaluation of the gradient 
- evaluation of the Gauss-Newton approximation of the Hessian
=#

mutable struct AugmentedLagrangian{T} where T
    residuals::Function
    nlconstraints::Function
    mu::T
end

function al_obj(x::Vector{T}, 
    y::Vector{T}, 
    al::AugmentedLagrangian{T}) where T 

    rx, cx = al.residuals(x), al.nlconstraints
    return 0.5*dot(rx,rx) + dot(y,cx) + 0.5*mu*dot(cx,cx)
end 

function al_gradient(x::Vector{T}, 
    al::AugmentedLagrangian{T},
    jac_res::Function, 
    jac_nlcons::Function)

    rx,cx  = al.residuals(x), al.nlconstraints(x)
    Jx, Cx  = jac_res(x), jac_nlcons(x)
    y_bar = y + al.mu*cx

    return Jx'*rx + Cx'*y_bar
end

function al_gn_hessian(x::Vector{T}, 
    al::AugmentedLagrangian{T},
    jac_res::Function, 
    jac_nlcons::Function)

    Jx, Cx  = jac_res(x), jac_nlcons(x)

    return Jx'*Jx + al.mu*Cx'*Cx
end

#### The solver 

### Constraints relatite methods

function is_feasible(x::Vector{T}, A::Matrix{T}, x_l::Vector{T}, x_u::Vector{T})

    return isapprox(A*x,b) && all(x_l .<= x) && all(x .<= x_u)
end

### Tolerances relative methods
function initial_tolerances(mu::T,
    omega0::T,
    eta0::T,
    k_crit::T,
    k_feas::T,)

    omega = omega0 * mu ^ (-k_crit)
    eta = eta0 * mu ^ (-k_feas)
    return omega, eta
end

#= TRALCNLSS stands for Trust Region Augmented Lagrangian Constrainted Nonlinear Least Squares Solver =#

function tralclss(x0::Vector{T},
    y0::Vector{T},
    residuals::Function,
    jac_res::Function,
    nlconstraints::Function,
    jac_nlcons::Function,
    A::Matrix{T},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T};
    mu0::T = T(10),
    tau::T = T(100),
    omega0::T = T(1),
    eta0::T = T(1),
    k_crit::T = T(1),
    k_feas::T = T(0.1),
    beta_crit::T = T(1),
    beta_feas::T = 0.9) where T

    # Initialize tolerances
    omega, eta = initial_tolerances(mu0, omega0, eta0, k_crit, k_feas)

    # Instantiate the augmented Lagrangian
    aug_lag = AugmentedLagrangian(residuals, nlconstraints, mu0)
    
    return
end


#= Solves the sub problem of an outer iteration
Approximately minimize the Augmented Lagrangian function with respect to the primal variable with tolerance ω > 0=#
function inner_iteration(x0::Vector{T},
                         ) where T
    return
end

#= Return the norm of the reduced gradient 'Ñᵀg'
    
    * 'g' is the gradient of the augmented Lagrangian at current iterate

    * 'Ñ' is an orthonormal matrix representing the null space of current active constraints

=#

function norm_reduced_gradient(
    g::Vector{T},
    A::Matrix{T},
    fix_bounds::BitVector,
    chol_aug_aat::Cholesky{T,Matrix{T}}) where T

    reduced_g = projection(A,chol_aug_aat,fix_bounds,g)
    return norm(reduced_g)
end

function norm_reduced_gradient(
    g::Vector{T},
    A::Matrix{T},
    chol_aat::Cholesky{T,Matrix{T}}) where T

    reduced_g = projection(A,chol_aat,g)
    return norm(reduced_g)
end
# Method the case with no active bounds


#= Compute an approximate Cauchy point by finding the first local minimum of a piecewise quadratic path 

DOES NOT WORK 
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

=#
