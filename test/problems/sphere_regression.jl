#= Test problem generated by Claude 3 
    - 3 paramaters 
    - 4 residuals 
    - 1 nonlinear equality constraints
    - 1 linear equality constraint 
    - lower and upper bounds on all variables 
    =#

# Linear constraints 
x_l = [-2., -1.5, 0]
x_u = [2., 1.5, 2.]
A = [1. 2 -1]
b = [0.5]

# Residuals 

r(x::Vector) = [x[1]^2 + x[2]^2 - 2*x[1] + sin(x[1]+x[2]) - 1.5, 
    x[1]*x[2] + 0.5*cos(2*x[1]) - 0.8, 
    (x[1]-1.0)^2 + (x[2]-0.5)^2 - x[3], 
    x[3]^2-x[1]+0.3*sin(x[3])-0.2]

jac_r(x::Vector) = [2*x[1]-2+cos(x[1]+x[2]) 2*x[2]+cos(x[1]+x[2]) 0.0;
x[2]-sin(2*x[1]) x[1] 0.0;
2*(x[1]-1) 2*(x[2]-0.5) -1.0;
-1.0 0.0 2*x[3]+0.3*cos(x[3])]
# Nonlinear constraints 
c(x::Vector) = [x[1]^2 + x[2]^2 + x[3]^2 - 3]

jac_c(x::Vector) = [2*x[1] 2*x[2] 2*x[3]]

# Starting point 
x0 = [1.0, 0.5, 1.5]

verbose = false

@testset "Sphere regression problem" begin
   
    x_sol, y_sol = tralcnllss(x0,
    r,
    jac_r,
    c,
    jac_c,
    A,
    b,
    x_l,
    x_u;
    max_outer_iter=100,
    max_inner_iter=250)

    verbose && @show x_sol
    verbose && @show y_sol
    rx_sol = r(x_sol)
    Jx_sol = jac_r(x_sol)
    Cx_sol = jac_c(x_sol)
    cx_sol = c(x_sol)
    verbose && @show cx_sol
    ∇lag = Jx_sol'*rx_sol + Cx_sol'*y_sol
    p_xm∇lag = BEnlsip.projection_polyhedron(x_sol-∇lag,A,b,x_l,x_u)
    opt_measure = norm(x_sol-p_xm∇lag)
    verbose && @show opt_measure
    feas_tol = sqrt(eps(Float64))
    opt_tol = 1e-7
    @test norm(cx_sol) < feas_tol
    @test BEnlsip.is_feasible(x_sol,A,x_l,x_u;b=b)
    @test opt_measure < opt_tol
end