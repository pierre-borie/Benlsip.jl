@testset "Projection on polyhedral sets" begin
    
    # Feasibe set of problem 48 from Hock Schittkowski collection
    
    (m,n) = (1,5)
    A = [1.0 1 1 1 1;
    0 0 1 -2 -2]
    b = [5.0, -3]
    x_hs = [3.0, 5, -3, 2, -2]

    fixed_i = [1,2]
    ifix = BitVector([i in fixed_i for i=1:n])
    Z = Matrix{Float64}(I,n,n)[ifix,:]
    # Z = [1.0 0 0 0 0;
    # 0 1 0 0 0]
    @show Z
    B = vcat(A,Z)

    v = zeros(n)
    chol_aa = cholesky(A*A')
    chol_bb = BEnlsip.cholesky_aug_aat(A,ifix, chol_aa)


    BEnlsip.projection!(v,A,chol_bb,ifix,x_hs)

    proj_xhs = [0.0, 0, 0, 2, -2] # obtained with ipopt

    Av = A*v
    @show A*v
    
    @test B*x_hs ≈ BEnlsip.mul_a_tilde(A,ifix,x_hs)
    # Feasibility
    @test all(≤(eps()), v[ifix])
    @test dot(Av, Av) ≤ eps() 
    # minimality
    @test v ≈ proj_xhs

end