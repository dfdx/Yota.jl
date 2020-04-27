using LinearAlgebra
using Distributions


function mvnormal_loss(mu, s, x)
    d = MvNormal(mu, s)
    return logpdf(d, x)
end

@testset "distributions" begin
    # diagonal covariance matrix
    # values for comparison obtained from PyTorch
    mu = zeros(2)
    s = [1.0 0.0; 0.0 1.0]
    x = [-0.5, 0.5]
    _, g = grad(mvnormal_loss, mu, s, x)
    @test g[1] == [-0.5, 0.5]
    @test g[2] == [-0.375 -0.125; -0.125 -0.375]
    @test g[3] == [0.5, -0.5]
    # full covariance matrix
    mu = zeros(2)
    s = [1.0 0.5; 0.5 1.0]
    x = [-0.5, 0.5]
    _, g = grad(mvnormal_loss, mu, s, x)
    @test isapprox(g[1], [-1, 1])
    @test isapprox(g[2], [-0.166667 -0.166667; -0.166667 -0.166667], rtol=1e-5)
    @test isapprox(g[3], [1, -1])
end
