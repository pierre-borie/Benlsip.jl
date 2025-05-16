"""
    new_point(x,y,mu,residuals,nlconstraints,jac_res,jac_nlcons)

Methods 'new_point' evaluate at 'x' and return the following 

* residuals 'rx', nonlinear constraints 'cx'

* Respectice Jacobians of the residuals and constraints 'Jx', 'Cx'

* first-order estimate of the multipliers 'y_bar'
"""

function new_point(
    x::Vector{T},
    y::Vector{T},
    mu::T,
    residuals::Function,
    nlconstraints::Function,
    jac_res::Function,
    jac_nlcons::Function)

    rx, cx = residuals(x), nlconstraints(x)
    Jx, Cx = jac_res(x), jac_nlcons(x)
    y_bar = y + mu*cx 
    return rx, cx, Jx, Cx, y_bar
end

""" s_inner_hs(s,mu,J,C)

Evaluate at 's' the quadratic term 'sᵀHs' where 'H = JᵀJ + μCᵀC' is the Gauss-Newton approximation of the augmented Lagrangian Hessian.
"""
function s_inner_hs(
    s::Vector{T},
    mu::T,
    J::Matrix{T}, 
    C::Matrix{T}) where T

    Js = J*s
    Cs = C*s 
    return dot(Js,Js) + mu*dot(Cs,Cs)
end

""" hs(s,mu,J,C)

Evaluate at 's' the matrix-vector product 'Hs' where 'H = JᵀJ + μCᵀC' is the Gauss-Newton approximation of the augmented Lagrangian Hessian.
"""
function hs(
    s::Vector,
    mu::T,
    J::Matrix{T},
    C::Matrix{T})

    Js, muCs = J*s, mu*C*s 
    return J'*Js + C'*muCs
end
#### The solver 

### Constraints relatite methods

function is_feasible(
    x::Vector{T}, 
    A::Matrix{T}, 
    x_l::Vector{T}, 
    x_u::Vector{T};
    b::Vector{T}=zeros(T,size(A,2))) where T

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
function solve_subproblem(x0::Vector{T},
    y::Vector{T},
    mu::T,
    residuals::Function,
    nlconstraints::Function,
    jac_res::Function,
    jac_nlcons::Function,
    A::Matrix{T},
    chol_aat::Cholesky{T,Matrix{T}},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T},
    delta0::T,
    omega_tol::T;
    eta1::T,
    eta2::T,
    gamma1::T,
    gamma2::T,
    gamma_c::T = T(10),
    kappa0::T = T(1e-2)) where T


    @assert (0 < eta1 <= eta2 < 1) && (0 < gamma1 <= gamma2) "Invalid trust region updates paramaters"

    rk, ck, Jk, Ck, yk_bar = new_point(x0, y, mu, residuals, nlconstraints, jac_res, jac_nlcons) 

    mk = 0.5*dot(rk,rk) * 0.5*mu*dot(ck,ck) # objective function
    gk = Jk'*rk + Ck'*yk_bar # gradient

    terminated = false
    return
end

function cauchy_step(
    x::Vector{T},
    g::Vector{T},
    J::Matrix{T},
    C::Matrix{T},
    mu::T,
    A::Matrix{T},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T},
    delta::T,
    t0::T,
    kappa0::T,
    gamma_c::T) where T

    # Bounds on the step 
    t_trial = t0
    s = projection_polyhedron(x-t_trial*g, A, b, x_l, x_u) - x

    increase = false
    s_infnorm = norm(Inf,s)
    if s_infnorm > delta
        increase = false
    else
        gts = dot(g,s)
        qs = 0.5*s_inner_hs(s,mu,J,C) + gts
        progress = qs <= kappa0 * gts
    end

    if increase
        t_c = t_trial
        progress = true
        while progress
            t_trial *= gamma_c
            s = projection_polyhedron(x-t_trial*g, A, b, x_l, x_u) - x
            s_infnorm = norm(Inf,s)
            if s_infnorm <= delta
                gts = dot(g,s)
                qs = 0.5*s_inner_hs(s,mu,J,C) + gts
                progress = qs <= kappa0 * gts
                if progress t_c = t_trial end 
            else
                progress = false
            end
        end
    else
        satisfied = false
        while !satisfied
            t_trial /= gamma_c
            s =  projection_polyhedron(x-t_trial*g, A, b, x_l, x_u) - x
            s_infnorm = norm(Inf,s)
            if s_infnorm <= delta
                gts = dot(g,s)
                qs = 0.5*s_inner_hs(s,mu,J,C) + gts
                satisfied = qs <= kappa0 * gts
            end
        end
        t_c = t_trial
    end

    # This computation is (likely) uneccessary, retrieving the cauchy step from above should be possible 
    s_c = projection_polyhedron(x-t*g, A, b, x_l, x_u) - x
    return s_c
end

function minor_iterate(
    x::Vector{T},
    g::Vector{T},
    J::Matrix{T},
    C::Matrix{T},
    mu::T,
    A::Matrix{T},
    x_l::Vector{T},
    x_u::Vector{T},
    fixed_var::BitVector,
    delta::T,
    kappa2::T) where T

    n = size(x,1)

    free_var = free_index(fixed_var)
    w_u, w_l = fill(Inf,n), fill(-Inf,n)

    w_u[free_var] .= (t -> min(t, delta)).(x_u[free_var]-x[free_var])
    w_l[free_var] .= (t -> max(t, -delta)).(x_l[free_var]-x[free_var])
end

#= Return the norm of the reduced gradient 'Ñᵀg'
    
    * 'g' is the gradient of the augmented Lagrangian at current iterate

    * 'Ñ' is an orthonormal matrix representing the null space of current active linear constraints

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
