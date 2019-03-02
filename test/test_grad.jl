import Statistics

loss_simple(W, b, x) = sum(W * x .+ b)
loss_double_broadcast(W, b, x) = sum(sin.(W * x) .+ b)
loss_kw_mean(W, b, x) = Statistics.mean(W * x .+ b; dims=1)[1]


@testset "grad: basic" begin
    args = (rand(3, 4), rand(3), rand(4))
    @test gradcheck(loss_simple, args...)
    @test gradcheck(loss_double_broadcast, args...)

    val, g = grad(loss_kw_mean, args...)
    @test val == loss_kw_mean(args...)
    @test any(op isa Call && op.fn == mean_grad for op in g.tape)
    @test gradcheck(loss_kw_mean, args...)
end


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
        val, g = grad(sum_bcast, args...)
        for i=1:length(args)
            @test size(g[i]) == size(args[i])
        end
    end
end


@testset "grad: transpose" begin
    a = rand(5)
    b = rand(2, 3)
    @test grad(x -> sum(transpose(x)), a)[2][1] == ones(size(a))
    @test grad(x -> sum(adjoint(x)), a)[2][1] == ones(size(a))
    @test grad(x -> sum(transpose(x)), b)[2][1] == ones(size(b))
    @test grad(x -> sum(adjoint(x)), b)[2][1] == ones(size(b))
end


mutable struct Linear{T}
    W::AbstractArray{T,2}
    b::AbstractArray{T}
end

forward(m::Linear, X) = m.W * X .+ m.b

loss(m::Linear, X) = sum(forward(m, X))

@testset "grad: structs" begin

    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]
    val, g = grad(loss, args...)

    @test val == loss(args...)

    play!(g.tape)
    val, g = grad(loss, args...)
    @test val == loss(args...)

end


# HESS = randn(3,3)
# hessian_fun(x) = x'*(HESS*x)
# hessian_fun2(x) = 0.5*x'*(HESS*x)

# @testset "grad: adjoint" begin
#     H = randn(3,3)
#     x = rand(3)
#     val, g = grad(hessian_fun2, x)
#     @test val isa Real
# end


@testset "grad: compiled" begin
    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]

    val, g = grad(loss, args...)

    # compiled vs. non-compiled
    tape = g.tape
    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]

    val1 = play!(tape, args...; use_compiled=true)
    last_val1 = tape[end].val
    val2 = play!(tape, args...; use_compiled=false)
    last_val2 = tape[end].val
    @test val1 == val2
    @test last_val1 == last_val2

    # # (* -> mul!) for mixed array-scalar vars
    # x = rand(3)
    # val1, g1 = grad(hessian_fun, x)  # interpreted
    # val2, g2 = grad(hessian_fun, x)  # compiled
    # @test val1 == val2
    # @test g1[1] == g2[1]
end
