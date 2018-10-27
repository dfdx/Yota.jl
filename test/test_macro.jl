
relu(x::AbstractArray) = max.(x, 0)
relu_grad(x::AbstractArray) = (x .> 0)

@primitive relu(x::AbstractArray)
@primitive relu_grad(x::AbstractArray)

@grad relu(x::AbstractArray) 1 relu_grad(x)

@testset "macro" begin

    tape = Tape()
    x = tracked(tape, rand(10))
    y = relu(x)

    @test length(tape) == 1
    @test tape[1].val == x.val


end
