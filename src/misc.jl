function print_tralcnllss_header(
    n::Int,
    d::Int,
    p::Int,
    m::Int,
    x_l::Vector{T},
    x_u::Vector{T},
    crit_tol::T,
    feas_tol::T,
    tau::T,
    eta1::T,
    eta2::T,
    gamma1::T,
    gamma2::T;
    io::IO=stdout) where T

    print(io,"\n\n")
    println(io, '*'^64)
    println(io, "*",' '^62,"*")

    println(io, "*"," "^23,"BEnlsip.jl v-DEV"," "^23,"*")
    println(io, "*",' '^62,"*")
    println(io, "*                   Better version of ENLSIP                   *")
    println(io, "*",' '^62,"*")
    println(io, '*'^64)

    println(io, "\nProblem dimensions")
    println(io, "Number of parameters.................: ", @sprintf("%5i", n))
    println(io, "Number of residuals..................: ", @sprintf("%5i", d))
    println(io, "Number of nonlinear constraints......: ", @sprintf("%5i", p))
    println(io, "Number of linear constraints.........: ", @sprintf("%5i", m))
    println(io, "Number of lower bounds...............: ", @sprintf("%5i", count(isfinite, x_l)))
    println(io, "Number of upper bounds...............: ", @sprintf("%5i", count(isfinite, x_u)))
    println(io, "\nAlgorithm parameters")
    println(io, "Optimality tolerance.................................:", @sprintf("%.6e", crit_tol))
    println(io, "Nonlinear constraints feasibility tolerance..........:", @sprintf("%.6e", feas_tol))
    println(io, "Increase penalty parameter factor....................:", @sprintf("%5f", tau))
    println(io, "Step acceptance treshold.............................:", @sprintf("%5f", eta1))
    println(io, "Great step acceptance treshold.......................:", @sprintf("%5f", eta2))
    println(io, "Trust region increase factor.........................:", @sprintf("%5f", gamma2))
    println(io, "Trust region decrease factor.........................:", @sprintf("%5f", gamma1))
    println(io,"\n\n")

    return
end

function print_outer_iter_header(
    k::Int,
    objective::T,
    nl_feas::T,
    mu::T,
    pix::T,
    omega::T;
    io::IO=stdout,
    first::Bool=false) where T

    println(io,"\n",'='^80)
    println(io,"                          Outer iter $k")
    println(io,"  objective    nl feasibility     μ      criticality   tolerance")
    if first
        @printf(io, "%.7e   %.6e  %.2e        -         %.2e", objective, nl_feas, mu, omega)
    else
        @printf(io, "%.7e   %.6e  %.2e     %.2e     %.2e", objective, nl_feas, mu, pix, omega)
    end
    println(io,"\n",'='^80)
    println(io,"iter     AL value       ||s||        Δ          ρ")
    return
end

function print_inner_iter(
    k::Int,
    obj::T,
    norm_step::T,
    radius::T,
    rho::T;
    io::IO=stdout) where T

    @printf(io, "%4d   %.6e   %.2e   %.2e   %.2e\n", k, obj, norm_step, radius, rho)
    return
end
