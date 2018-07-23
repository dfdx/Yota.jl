mutable struct Linear{T}
    W::AbstractArray{T,2}
    b::AbstractArray{T}
end

forward(m::Linear, X) = m.W * X

loss(m::Linear, X) = sum(forward(m, X))


@testset "grad: structs" begin

    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]
    val, g = grad(loss, args...)

    @test val == loss(args...)

    play!(g.tape)
    val, g = grad(loss, args...)
    @test val == loss(args...)
    
end
