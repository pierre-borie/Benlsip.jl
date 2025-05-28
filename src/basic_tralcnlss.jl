abstract type TralcnllsData end

mutable struct AlHessian{T<:Real} <: TralcnllsData
    J::Matrix{T}
    C::Matrix{T}
    mu::T 
end

@enum CG_status solved bound_hit negative_curvature max_iter_reached

"""
    new_point(x,y,mu,residuals,nlconstraints,jac_res,jac_nlcons)

Methods 'new_point' evaluate at 'x' and return the following 

* residuals 'rx'

* nonlinear constraints 'cx' 

* first-order estimate of the multipliers 'y_bar'

* value of the augmented Lagrangian function 'mx'

* gradient 'g' of the augmented Lagrangian

* Gauss-Newton approximmation of the Hessian 'H' of the augmented Lagrangian encoded in [`AlHessian`](@ref) type
"""

function new_point(
    x::Vector{T},
    y::Vector{T},
    mu::T,
    residuals::F,
    nlconstraints::F,
    jac_res::F,
    jac_nlcons::F) where {T, F<:Function}

    rx, cx = residuals(x), nlconstraints(x)
    Jx, Cx = jac_res(x), jac_nlcons(x)
    y_bar = y + mu*cx 
    mx = 0.5*dot(rx,rx) * 0.5*mu*dot(cx,cx) # objective function
    g = Jx'*rx + Cx'*y_bar # gradient
    H = AlHessian(Jx,Cx,mu) # Hessian

    return rx, cx, y_bar, mx, g, H
end

function evaluate_al(
    x::Vector{T},
    mu::T,
    residuals::F,
    nlconstraints::F) where {T, F<:Function}

    rx, cx = residuals(x), nlconstraints(x)
    mx = 0.5*dot(rx,rx) * 0.5*mu*dot(cx,cx) # objective function
    return rx, cx, mx
end

function first_derivatives(
    x::Vector{T},
    y::Vector{T},
    mu::T,
    rx::Vector{T},
    cx::Vector{T},
    jac_res::F,
    jac_nlcons::F) where {T, F<:Function}

    Jx, Cx = jac_res(x), jac_nlcons(x)
    y_bar = y + mu*cx # first-order multipliers estimates
    g = Jx'*rx + Cx'*y_bar # gradient

    return y_bar, Jx, Cx, g
end

function second_derivatives(
    Jx::Matrix{T},
    Cx::Matrix{T},
    mu::T) where T 

    return AlHessian(Jx,Cx,mu)
end

""" vthv(H,v)

The quadratic term 'vᵀHv' where 'H = JᵀJ + μCᵀC' is the Gauss-Newton approximation of the augmented Lagrangian Hessian encoded into
the [`AlHessian`](@ref) type.
"""
function vthv(H::AlHessian{T}, v::Vector{T}) where T
    Jv = H.J*v
    Cv = H.C*v 
    return dot(Jv,Jv) + H.mu*dot(Cv,Cv)
end 

""" Base.:*(H::AlHessian, v::Vector)

Overload the '*' operator to be compatible with the type [`AlHessian`](@ref).
"""
function Base.:*(H::AlHessian{T}, v::Vector{T}) where T
    Jv = H.J*v 
    muCv = H.mu*H.C*v 
    return H.J' * Jv + H.C' * muCv
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
    C::Matrix{T}) where T

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
function initial_tolerances(
    mu::T,
    omega0::T,
    eta0::T,
    k_crit::T,
    k_feas::T) where T

    omega = omega0 / (mu^k_crit)
    eta = eta0 / (mu^k_feas)
    return omega, eta
end

#= TRALCNLSS stands for Trust Region Augmented Lagrangian Constrainted Nonlinear Least Squares Solver =#

function tralcllss(
    x0::Vector{T},
    y0::Vector{T},
    residuals::F,
    jac_res::F,
    nlconstraints::F,
    jac_nlcons::F,
    A::Matrix{T},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T};
    mu0::T = T(10),
    tau::T = T(100),
    omega0::T = T(1),
    eta0::T = T(1),
    feas_tol::T,
    crit_tol::T,
    k_crit::T = T(1),
    k_feas::T = T(0.1),
    beta_crit::T = T(1),
    beta_feas::T = T(0.9),
    eta1::T,
    eta2::T,
    gamma1::T,
    gamma2::T,
    gamma_c::T,
    kappa0::T,
    kappa2::T,
    kappa3::T,
    max_outer_iter::Int=500,
    max_inner_iter::Int=500,
    max_minor_iter::Int=50) where {T<:Real, F<:Function}

    # Sanity check
    @assert (0 < eta1 <= eta2 < 1) && (0 < gamma1 < 1 < gamma2) "Invalid trust region updates paramaters"

    # Initializations
    n = size(x0)
    chol_aat = cholesky_aat(A) # Cholesky decomposition of AAᵀ
    x = Vector{T}(undef,n)
    x[:] = x0[:]
    cx = nlconstraints(x)
    mu = mu0


    omega, eta = initial_tolerances(mu0, omega0, eta0, k_crit, k_feas) # tolerances 
    y = least_squares_multipliers(x,residuals,jac_res,jac_nlcons) # Initial Lagrange multipliers 

    first_order_critical = false
    outer_iter = 1

    while !first_order_critical && outer_iter <= max_outer_iter

        x_next, cx_next, pix = solve_subproblem(x,
        y,
        mu,
        residuals,
        nlconstraints,
        jac_res,
        jac_nlcons,
        A,
        chol_aat,
        b,
        x_l,
        x_u,
        max_minor_iter,
        max_inner_iter,
        omega,
        eta1,
        eta2,
        gamma1,
        gamma2,
        gamma_c,
        kappa0,
        kappa2,
        kappa3)

        feas_measure = norm(cx_next)

        if feas_measure <= eta
            x .= x_next
            cx .= cx_next
            first_order_critical = pix <= crit_tol && feas_measure <= feas_tol

            if !first_order_critical
                # Update the iterate, multipliers and decrease tolerances (penalty parameter is unchanged)
                y = first_order_multipliers(y,cx,mu)
                omega /= mu^(beta_crit)
                eta /= mu^(beta_feas)
            end
        else
            # Increase the penalty parameter lesser decrease of the tolerances,  (iterate and multipliers are unchanged)
            mu *= tau
            omega = omega0 / (mu^k_crit)
            eta = eta0 / (mu^k_feas)   
        end

        outer_iter += 1

    end

    
    
    
    return x, y
end


#= Solves the sub problem of an outer iteration
Approximately minimize the Augmented Lagrangian function with respect to the primal variable with tolerance ω > 0=#
function solve_subproblem(x0::Vector{T},
    y::Vector{T},
    mu::T,
    residuals::F,
    nlconstraints::F,
    jac_res::F,
    jac_nlcons::F,
    A::Matrix{T},
    chol_aat::Cholesky{T,Matrix{T}},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T},
    nb_minor_step::Int,
    k_max::Int,
    omega_tol::T,
    eta1::T,
    eta2::T,
    gamma1::T,
    gamma2::T,
    gamma_c::T,
    kappa0::T,
    kappa2::T,
    kappa3::T) where {T<:Real, F<:Function}


    

    # Dimensions
    (_,n) = size(A)
    x = Vector{T}(undef,n)
    x[:] = x0[:]
    rx, cx, y_bar, mx, g, H = new_point(x0, y, mu, residuals, nlconstraints, jac_res, jac_nlcons) 

    
    t = T(1)
    delta = initial_tr(g)
    k = 1
    solved = false

    while !solved && k <= k_max
        # step and model reduction
        s, t, pred, fix_bounds, chol_aug_aat = inner_step(x,g,H,A,chol_aat,b,x_l,x_u,delta,t,nb_minor_step,kappa0,kappa2,kappa3,gamma_c)

        x_next = x+s
        rx_next, cx_next, mx_next = evaluate_al(x,mu,residuals,nlconstraints)
        ared = mx_next - mx

        rho = ared / pred

        if rho > eta1
            x .= x_next
            rx[:], cx[:], mx[:] = rx_next[:], cx_next[:], mx_next[:]
            y_bar, J, C, g = first_derivatives(x,y,mu,rx,cx,jac_res,jac_nlcons)
            H = second_derivatives(J,C,mu)
        end

        # Update trust region radius
        delta = update_tr(delta, rho, eta1, eta2, gamma1, gamma2)

        # Compute criticality measure
        pix = criticality_measure(g,fix_bounds,A,chol_aug_aat)
        
        # Termination criteria 
        solved = pix < omega_tol
        k += 1
    end

    return x, cx, pix
end

""" inner_step(x,g,H,A,chol_AAᵀ,b,xₗ,xᵤ,Δ,nb_minor_step,κ₀,κ₂,κ₃,γc)

Starting from the current iterate 'x', compute a step 's' such that the inner step 'x+s' sufficiently reduces the model.

On return 

* the step 's'

* the scalar 't_c' used to compute the Cauchy point 

* the reduction predicted by the model with step 's'

* indices of acitve bounds encoded in a 'BitVector' and the associated Cholesky decomposition
"""
function inner_step(
    x::Vector{T},
    g::Vector{T},
    H::AlHessian{T},
    A::Matrix{T},
    chol_aat::Cholesky{T,Matrix{T}},
    b::Vector{T},
    x_l::Vector{T},
    x_u::Vector{T},
    delta::T,
    t0::T,
    nb_minor_step::Int,
    kappa0::T,
    kappa2::T,
    kappa3::T,
    gamma_c::T) where T

    s, t_c = cauchy_step(x,g,H,A,b,x_l,x_u,delta,t0,kappa0,gamma_c)
    g_minor = H*s+g
    fix_bounds, chol_aug_aat = active_w_chol(s,x,x_l,x_u,delta,A,chol_aat)

    j = 1 # minor iterations index

    norm_reduced_g = norm_reduced_gradient(g,A,fix_bounds,chol_aug_aat)
    norm_reduced_g_minor =  norm_reduced_gradient(g,A,fix_bounds,chol_aug_aat)

    approx_solved = norm_reduced_g_minor <= kappa3 * norm_reduced_g

    cg_stop  = false

    # Minor iterates loop
    while j <= nb_minor_step && !approx_solved && !cg_stop

        # descent direction and termination status of the cg iterations
        w, cg_status = minor_iterate(x,s,g_minor,H,A,x_l,x_u,chol_aug_aat,fix_bounds,delta,kappa2)
        s .+= w # cumulated step 
        g_minor = H*s+g
        fix_bounds, chol_aug_aat = active_w_chol(s,x,x_l,x_u,delta,A,chol_aat) # New active set

        # Loop termination criteria
        norm_reduced_g = norm_reduced_gradient(g,A,fix_bounds,chol_aug_aat)
        norm_reduced_g_minor =  norm_reduced_gradient(g,A,fix_bounds,chol_aug_aat)
        approx_solved = norm_reduced_g_minor <= kappa3 * norm_reduced_g
        cg_stop = cg_status == negative_curvature
        j += 1
    end

    # Evaluate the reduction of the quadratic model 
    model_reduction = dot(s,g_minor)
    return s, t_c, model_reduction, fix_bounds, chol_aug_aat
end

function cauchy_step(
    x::Vector{T},
    g::Vector{T},
    H::AlHessian{T},
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
        # qs = 0.5*s_inner_hs(s,mu,J,C) + gts
        qs = 0.5*vthv(H,s) +  gts
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
                # qs = 0.5*s_inner_hs(s,mu,J,C) + gts
                qs = 0.5*vthv(H,s) +  gts
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
                # qs = 0.5*s_inner_hs(s,mu,J,C) + gts
                qs = 0.5*vthv(H,s) +  gts
                satisfied = qs <= kappa0 * gts
            end
        end
        t_c = t_trial
    end

    # This computation is (likely) uneccessary, retrieving the cauchy step from above should be possible 
    s_c = projection_polyhedron(x-t_c*g, A, b, x_l, x_u) - x
    return s_c, t_c
end

""" minor_iterate(x,s,g,H,A,xₗ,xᵤ,fixed_var,Δ,κ₂)

Compute a search direction 'w' and a steplength 'α' such that the next minor iterate 'x + s + α*w' provides a sufficient reduction, where

* 'x' is the current iterate 

* 's' is the previous minor step or, equivalently, 'x + s' is the previous minor iterate
"""
function minor_iterate(
    x::Vector{T},
    s::Vector{T},
    g_model::Vector{T},
    H::AlHessian{T},
    A::Matrix{T},
    x_l::Vector{T},
    x_u::Vector{T},
    chol_aug_aat::Cholesky{T,Matrix{T}},
    fixed_var::BitVector,
    delta::T,
    kappa2::T) where T

    n = size(x,1)

    x_minor = x+s
    # g_model = H*s + g
    free_var = free_index(fixed_var)
    w_u, w_l = fill(Inf,n), fill(-Inf,n)

    w_u[free_var] .= (t -> min(t, delta)).(x_u[free_var]-x_minor[free_var])
    w_l[free_var] .= (t -> max(t, -delta)).(x_l[free_var]-x_minor[free_var])

    w, cg_status = projected_cg(g_model, H, A, w_l, w_u, chol_aug_aat, fixed_var, kappa2)

    if cg_status != negative_curvature
        alpha = linesearch(g_model, H, w, w_l, w_u, fixed_var)
        w .+= alpha * w
    end

    return w, cg_status
end

""" projected_cg(r0,H,A,wₗ,wᵤ,chol_AₓAₓᵀ,fix_bounds,κ₂,atol)

Approximately solve the sub problem 

min 0.5 wᵀHw + wᵀr0 
s.t. Aw = 0
     wᵢ = 0,    i ∈ fix_bounds
     wₗ ≤ w ≤ wᵤ

with respect to 'w' and using the projected conjugate gradient method

Returns the obtained descent direction and the termination status, encoded in the 'Enum' [`CG_status`](@ref)
"""

function projected_cg(
    r0::Vector{T},
    H::AlHessian{T},
    A::Matrix{T},
    w_l::Vector{T},
    w_u::Vector{T},
    chol_aug_aat::Cholesky{T,Matrix{T}},
    fix_bounds::BitVector,
    kappa2::T;
    atol::T = sqrt(eps(T))) where T

    (m,n) = size(A)
    
    # Initialization
    w = zeros(n)
    # r = H*s + g
    r = r0
    v = projection(A,chol_aug_aat,fix_bounds,r)
    rtv = dot(r,v)
    p = -v

    tol_cg = kappa2 * norm(v)
    tol_negcurve = atol

    iter = 1
    max_iter = 2*(n-m-count(fix_bounds))
    approx_solved = false 
    neg_curvature = false
    outside_region = false

    while !(approx_solved || outside_region || neg_curvature || iter <= max_iter)
        Hp = H*p
        pHp = dot(p,Hp)

        if pHp <= tol_negcurve # negative curvature
            neg_curvature = true
            if abs(pHp) > tol_negcurve # nonzero curvature
                gamma = factor_to_boundary(p,w,w_l,w_u)
                w .+= gamma * p
            end
        else
            rtv = dot(r,v)
            alpha = rtv / pHp

            gamma = factor_to_boundary(p,w,w_l,w_u)
            outside_region = alpha > gamma
            if bound_hit
                w .+= gamma * p
            else 
                w .+= alpha*p
                r .+= alpha*Hp
                v = projection(A,chol_aug_aat,fix_bounds,r)
                rtv_next = dot(r,v)
                beta = rtv_next / rtv
                axpby!(-one(T), v, beta, p) # p = -v + βp
                rtv = rtv_next
                approx_solved = abs(rtv) < tol_cg
                iter += 1
            end
        end
    end

    status = if approx_solved
        solved
    elseif outside_region
        bound_hit
    elseif neg_curvature
        negative_curvature
    elseif iter == max_iter
        max_iter_reached
    end

    return w, status
end

function linesearch(
    g_model::Vector{T},
    H::AlHessian{T},
    w::Vector{T},
    w_l::Vector{T},
    w_u::Vector{T},
    fix_bounds::BitVector) where T 

    # unconstrained minimizer of the model along the descent direction
    wHw = vthv(H,w)
    alpha_opt = (wHw > 0 ? -dot(g_model,w) / wHw : Inf)

    # Maximum step allowed by the bounds on the free variables
    alpha_allowed =  Inf
    for i in axes(w,1)
        if !fix_bounds[i]
            if w[i] < 0
                alpha_allowed = min(alpha_allowed, w_l[i] / w[i])
            elseif w[i] > 0
                alpha_allowed = min(alpha_allowed, w_u[i] / w[i])
            end
        end
    end

    return min(alpha_opt, alpha_allowed)
end

function factor_to_boundary(
    p::Vector{T},
    w::Vector{T},
    w_l::Vector{T},
    w_u::Vector{T};
    atol::T = T(1e-10)) where T 

    gamma = Inf
    for i in axes(w,1)
        if p[i] <= -atol 
            gamma = min(gamma, (w_l[i] - w[i]) / p[i])
        elseif p[i] >= atol
            gamma = min(gamma, (w_u[i] - w[i]) / p[i])
        end
    end
    return gamma
end

""" initial_tr(g; factor)

Computes the initial trust region radius by taking a factor of the norm of 'g', the gradient of the model.

Returns 'factor * ||g||' with 'factor' default value set to '0.1'.
"""
function initial_tr(g::Vector{T};tr_factor::T=T(0.1)) where T
    return tr_factor * norm(g)
end

function update_tr(
    delta::T,
    rho::T,
    eta1::T,
    eta2::T,
    gamma1::T,
    gamma2::T) where T 

    delta_next = if rho > eta2 # very successful step
        gamma2 * delta 
    elseif rho < eta1 # bad step
        gamma1 * delta 
    else # successful step 
        delta 
    end
    
    return delta_next
end

function criticality_measure(
    g::Vector{T},
    fix_bounds::BitVector,
    A::Matrix{T},
    chol_aug_aat::Cholesky{T, Matrix{T}}) where T 

    if any(fix_bounds)
        crit = norm_reduced_gradient(g,A,fix_bounds,chol_aug_aat)
    else
        crit = norm_reduced_gradient(g,A,chol_aug_aat)
    end

    return crit
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

# TODO: implement a method to compute initial lagrange multipliers
function least_squares_multipliers(
    x::Vector{T},
    residuals::F,
    jac_res::F,
    jac_nlcons::F) where {T, F<:Function}
    
    g = jac_res(x)' * residuals(x) # gradient 
    C = jac_nlcons(x)
    chol_cct = cholesky(C*C') 
    y = begin   
        b = -C*g
        v = chol_cct.L \ b
        chol_cct.U \ v 
    end
    
    return y 
end

function first_order_multipliers(
    y::Vector{T},
    cx::Vector{T},
    mu::T) where T

    return y + mu*cx
end



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
