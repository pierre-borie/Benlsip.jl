@testset "Gauss-Newton Hessian structure test" begin
    
    n = 5

    J = rand(n,n)
    C = rand(n,n)
    mu = rand()
    v = rand(n)

    H = BEnlsip.AlHessian(J,C,mu)
    H_test = J'*J + mu*C'*C

    Hv_test = H_test*v
    @test H*v ≈ Hv_test
    @test BEnlsip.vthv(H,v) ≈ dot(v,Hv_test)
end

(m,n) = (3,6)

A = rand(m,n)
chol_aat = cholesky(A*A')
xlow = -rand(n)
xupp = rand(n) .+ 1
test_cons = BEnlsip.MixedConstraints(A,chol_aat; l=xlow,u=xupp)

act = [2,4,6]
test_cons.fixvars[act] .= true
update_chol!(test_cons, chol_aat)

B = vcat(A,Matrix{Float64}(I,n,n)[act,:])
greedy_L = cholesky(B*B').L
@testset "MixedConstraints structure" begin
    @test test_cons.fixvars == BitVector([i in act for i=1:n])
    @test test_cons.chol.L ≈ greedy_L
end

(m,n) = (2,5)
A = [1.0 1 1 1 1;
0 0 1 -2 -2]
chol_aat = cholesky(A*A')
b = [5.0, -3]
x_hs = [3.0, 5, -3, 2, -2]
proj_xhs = [0.0, 0, 0, 2, -2] # obtained with ipopt
fixed_i = [1,2]
ifix = BitVector([i in fixed_i for i=1:n])
B = vcat(A,Matrix{Float64}(I,n,n)[ifix,:])

@testset "Projection with MixedConstraints" begin
    test2_cons = BEnlsip.MixedConstraints(A,chol_aat,ifix)
    y = rand(m+count(ifix))
    @test B'*y ≈ BEnlsip.left_mul_tr(test2_cons, y)
    @test B*x_hs ≈ BEnlsip.left_mul(test2_cons, x_hs)

    proj_test = BEnlsip.projection(test2_cons,x_hs)
    normsq_proj = begin v = A*proj_test; dot(v,v) end
    @test all(<=(eps()), proj_test[ifix]) && normsq_proj <= eps()
    @test proj_xhs ≈ proj_test
end

(m,n) = (3,7)
A = rand(m,n)
chol_aat = cholesky(A*A')
xlow = -10 * ones(n)
xupp = 10 * ones(n)
test_cons = BEnlsip.MixedConstraints(A,chol_aat; l=xlow,u=xupp)

x = rand(n)
x[2] = -10.
indx = [3,5]
active = [2,3,5,7]
@testset "Active bounds identification and updating" begin
    BEnlsip.active_bounds!(test_cons,x,chol_aat)
    @test test_cons.fixvars[2] && !any(test_cons.fixvars[setdiff(1:n,2)])
    BEnlsip.add_active!(test_cons,chol_aat,indx)
    @test all(test_cons.fixvars[indx])
    BEnlsip.add_active!(test_cons,chol_aat,7)
    @test all(test_cons.fixvars[active]) && !any(test_cons.fixvars[setdiff(1:n,active)])
end
