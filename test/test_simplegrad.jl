simple_loss(W::AbstractMatrix, b::AbstractVector, x::AbstractArray) = sum(W * x .+ b)

@testset "simplegrad" begin
    W, b, x = rand(128, 784), rand(128), rand(784, 100)

    val1, g = grad(simple_loss, W, b, x)
    
    simple_loss_grad = simplegrad(simple_loss, W, b, x)    
    val2, dW, db, dx = simple_loss_grad(W, b, x)

    @test val1 == val2
    @test g[1] == dW
    @test g[2] == db
    @test g[3] == dx
end
