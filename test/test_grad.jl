
sum_bcast(x, y) = sum(x .+ y)

@testset "special bcast" begin
    for args in [
        (rand(3, 4), rand(3)),
        (rand(3, 4), rand(3, 1)),
        (rand(3, 4), rand(1, 4)),
        (rand(3), rand(3, 4)),
        (rand(3, 1), rand(3, 4)),
        (rand(1, 4), rand(3, 4)),
    ]
        val, g = grad(sum_bcast, args...; static=false)
        for i=1:length(args)
            @test size(g[i]) == size(args[i])
        end
    end
end



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

HESS = randn(3,3)
hessian_fun(x) = x'*(HESS*x)

@testset "grad: compiled" begin

    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]

    # static vs. dynamic
    val1, g1 = grad(loss, args...; static=false)
    grad(loss, args...; static=true)
    val2, g2 = grad(loss, args...; static=true)
    
    @test val1 == val2
    @test g1[1][(:W,)] == g2[1][(:W,)]

    # compiled vs. non-compiled
    tape = g1.tape
    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]
    
    play!(tape, args...; use_compiled=true)
    last_val1 = getvalue(tape[end])
    play!(tape, args...; use_compiled=false)
    last_val2 = getvalue(tape[end])

    @test last_val1 == last_val2

    # (* -> mul!) for mixed array-scalar vars    
    x = rand(3)
    val1, g1 = grad(hessian_fun, x)  # interpreted
    val2, g2 = grad(hessian_fun, x)  # compiled
    @test val1 == val2
    @test g1[1] == g2[1]
    
end

