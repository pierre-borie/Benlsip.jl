@testset "AlHessian structure test" begin
    
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

