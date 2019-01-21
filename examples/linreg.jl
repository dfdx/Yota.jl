# modified example from GradDescent.jl documentation
using GradDescent
using Yota
using Statistics
using Random


obj(Y, X, b) = mean((Y .- X * b) .^ 2.0) # objective to minimize

function main()
    Random.seed!(1) # set seed
    n = 1000 # number of observations
    d = 10   # number of covariates
    X = randn(n, d) # simulated covariates
    b = randn(d)    # generated coefficients
    ϵ = randn(n) * 0.1 # noise
    Y = X * b + ϵ # observed outcome
    
    epochs = 100 # number of epochs
    
    θ = randn(d) # initialize model parameters
    opt = Adam(α=1.0)  # initalize optimizer with learning rate 1.0
    
    for i in 1:epochs
        val, g = grad(obj, Y, X, θ)
        println("Epoch: $i; loss = $val")
        
        δ = update(opt, g[3])
        θ -= δ
    end
end
