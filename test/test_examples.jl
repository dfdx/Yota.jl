using Statistics
using Random


obj(Y, X, b) = mean((Y .- X * b) .^ 2.0) # objective to minimize

# modified example from GradDescent.jl documentation
@testset "linreg" begin

    srand(1) # set seed
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
