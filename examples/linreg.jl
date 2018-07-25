using Pkg

using GradDescent
using Yota
using Statistics
using Random


obj(Y, X, b) = mean((Y .- X * b) .^ 2.0) # objective to minimize


function main()

    srand(1) # set seed
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
        # here we use automatic differentiation to calculate
        # the gradient at a value
        # an analytically derived gradient is not required
        val, g = grad(obj, Y, X, θ)
        println(val)
        
        δ = update(opt, g[3])
        θ -= δ
    end

end


import Yota: TArray, Tape, tracked

function aux()
    tape = Tape()
    Y, X, b = [tracked(tape, v) for v in (Y, X, b)]
end
