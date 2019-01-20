using Statistics
using Random


obj(Y, X, b) = mean((Y .- X * b) .^ 2.0) # objective to minimize

# modified example from GradDescent.jl documentation
@testset "linreg" begin

    Random.seed!(1) # set seed
    n = 1000 # number of observations
    d = 10   # number of covariates
    X = randn(n, d) # simulated covariates
    b = randn(d)    # generated coefficients
    ϵ = randn(n) * 0.1 # noise
    Y = X * b + ϵ # observed outcome

    epochs = 100 # number of epochs

    θ = randn(d) # initialize model parameters

    loss_first, g = grad(obj, Y, X, θ)
    loss_last = loss_first
    for i in 1:epochs
        loss_last, g = grad(obj, Y, X, θ)
        # println("Epoch: $i; loss = $loss_last")
        δ = 0.01 * g[3]
        θ -= δ
    end
    @test loss_last < loss_first / 10   # loss reduced at least 10 times

end


mutable struct Linear2{T}
    W::AbstractMatrix{T}
    b::AbstractVector{T}
end

linear_loss(m::Linear2, X, y) = mean((m.W * X .+ m.b .- y) .^ 2.0)

@testset "linreg 2" begin

    n_vars = 10
    n_out = 2
    n_obs = 1000
    X = randn(n_vars, n_obs)
    W_true = rand(n_out, n_vars)
    b_true = rand(n_out)
    ϵ = randn(n_out, n_obs) * 0.1 # noise
    Y = W_true * X .+ b_true .+ ϵ

    m = Linear2(rand(n_out, n_vars), rand(n_out))

    epochs = 100

    for i in 1:epochs
        val, g = grad(linear_loss, m, X, Y)
        # println("Epoch: $i; loss = $val")
        update!(m, g[1], (x, gx) -> x - 0.1 * gx)
    end

    @test isapprox(m.W, W_true; atol=0.1)
    @test isapprox(m.b, b_true; atol=0.1)

end
