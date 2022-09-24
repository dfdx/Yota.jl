import Statistics
import Yota: isprimitive, ChainRulesCtx
import ChainRulesCore
import ChainRulesCore: rrule, Tangent, ZeroTangent, NoTangent, @opt_out


@testset "ChainRulesCtx" begin
    @test isprimitive(ChainRulesCtx(), sum, rand(3, 4))
    @test isprimitive(ChainRulesCtx(), Core.kwfunc(sum), (dims=1,), sum, rand(3, 4))
    @test !isprimitive(ChainRulesCtx(), Core.kwfunc(sum), (dims=1,), sum, rand())
end


loss_simple(W, b, x) = sum(W * x .+ b)
loss_double_broadcast(W, b, x) = sum(sin.(W * x) .+ b)
loss_double_broadcast2(b, x) = sum(x .* x .+ b)
loss_kw_mean(W, b, x) = Statistics.mean(W * x .+ b; dims=1)[1]


chain_foo(x::Number) = :ok

rrule(::typeof(chain_foo), ::Number) = ZeroTangent(), dy -> ZeroTangent()
rrule(::typeof(chain_foo), ::Real) = ZeroTangent(), dy -> ZeroTangent()
rrule(::typeof(chain_foo), ::Float64) = ZeroTangent(), dy -> ZeroTangent()
@opt_out rrule(::typeof(chain_foo), ::Real)


@testset "grad: simple" begin
    args3 = (rand(4, 3), rand(4), rand(3))
    args4 = (rand(4, 3), rand(4), rand(3), rand(4))
    @test gradcheck((W, b, x) -> sum(W * x .+ b), args3...)
    @test gradcheck((W, b, x) -> sum(tanh.(W * x .+ b)), args3...)
    @test gradcheck((W, b, x, y) -> sum(abs2.(tanh.(W * x .+ b) .- y)), args4...)
end


# originally a simplified version of lstm_forward
function multipath(y, c)
    f = tanh.(y[1:5, :])
    c_ = f .* c
    h_ = tanh.(c_)
    return h_, c_
end

@testset "grad: multipath" begin
    y = rand(20, 4)
    c = rand(5, 4)
    loss = (y, c) -> begin
            h, c = multipath(y, c)
            sum(h)
            end
    args = (y, c)
    @test gradcheck(loss, args...)
end


@testset "grad: kw" begin
    args = (rand(3, 4), rand(3), rand(4))
    @test gradcheck(loss_simple, args...)
    @test gradcheck(loss_double_broadcast, args...)
    @test gradcheck(loss_double_broadcast2, rand(3), rand(3))

    val, g = grad(loss_kw_mean, args...)
    @test val == loss_kw_mean(args...)
    @test gradcheck(loss_kw_mean, args...)

    @test gradcheck(x -> sum(sum(x, dims=1)), rand(2, 3))
end


@testset "grad: input-only" begin
    _, g = grad((x, y) -> x, 42.0, 54.0)
    @test g == (ZeroTangent(), 1, ZeroTangent())
    _, g = grad((x, y) -> y, 42.0, 54.0)
    @test g == (ZeroTangent(), ZeroTangent(), 1)
end


@testset "grad: constant" begin
    _, g = grad(x -> 1, 42.0)
    @test g == (ZeroTangent(), ZeroTangent())
    # _, g = grad(xs -> sum(map(_->1, xs)), [1.0, 2.0])
    # @test g == (ZeroTangent(), ZeroTangent())
end


@testset "grad: getindex" begin
    x = rand(3, 4, 5)
    x1 = zero(x); x1[1] = 1
    @test grad(x -> x[1], x)[2][2] == x1
    x2 = zero(x); x2[1, 2, 1] = 1
    @test grad(x -> x[1, 2, 1], x)[2][2] == x2
    x3 = zero(x); x3[:, 1, :] .= 1
    @test grad(x -> sum(x[:, 1, :]), x)[2][2] == x3
end


@testset "grad: cat" begin
    @test gradcheck((a, b) -> sum(vcat(a, b)), ones(2, 3), ones(3, 3))
    @test gradcheck((a, b) -> sum(hcat(a, b)), ones(2, 3), ones(2, 4))

    arrs = Any[i * ones(3, 4, i, 6) for i=1:4]
    @test gradcheck((a, b, c, d) -> sum(cat(a, b, c, d; dims=3)), arrs...)
end


@testset "grad: iterate" begin
    # iterate over tuple, e.g. for x in (1.0, 2.0, 3.0)
    x = (1.0, 2.0, 3.0)
    CT = Tangent{typeof(x)}
    @test grad(x -> iterate(x)[1], x)[2][2] == CT(1.0, ZeroTangent(), ZeroTangent())
    @test grad(x -> iterate(x, 2)[1], x)[2][2] == CT(ZeroTangent(), 1.0, ZeroTangent())
    @test grad(x -> iterate(x, 3)[1], x)[2][2] == CT(ZeroTangent(), ZeroTangent(), 1.0)

    # iterate over array, e.g. for x in [1.0, 2.0, 3.0]
    x = [1.0, 2.0, 3.0]
    # TODO (uncomment when scatter_add is fixed)
    # @test grad(x -> iterate(x)[1], x)[2][1] == [1.0, 0, 0]
    # @test grad(x -> iterate(x, 2)[1], x)[2][1] == [0, 1, 0]
    # @test grad(x -> iterate(x, 3)[1], x)[2][1] == [0, 0, 1.0]

    x = (1:3)
    @test grad(x -> iterate(x)[1], x)[2][2] == ZeroTangent()
    @test grad(x -> iterate(x, 1)[1], x)[2][2] == ZeroTangent()

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
        # (rand(3, 4), rand()),
        # (rand(), rand(3, 4)),
    ]
        @test gradcheck(sum_bcast, args...)
        val, g = grad(sum_bcast, args...)
        for i=1:length(args)
            @test size(g[i + 1]) == size(args[i])
        end
    end
end

sum_prod_bcast(x, y) = sum(x .* y)

@testset "special bcast 2" begin
    for args in [
        (rand(3, 4), rand(3)),
        (rand(3, 4), rand(3, 1)),
        (rand(3, 4), rand(1, 4)),
        (rand(3), rand(3, 4)),
        (rand(3, 1), rand(3, 4)),
        (rand(1, 4), rand(3, 4)),
        # (rand(3, 4), rand()),
        # (rand(), rand(3, 4)),
    ]
        @test gradcheck(sum_prod_bcast, args...)
        val, g = grad(sum_prod_bcast, args...)
        for i=1:length(args)
            @test size(g[i + 1]) == size(args[i])
        end
    end
end

@testset "grad: literal_pow" begin
    @test gradcheck(x -> sum(x .^ 2), rand(3, 4))
end

@testset "grad: transpose" begin
    a = rand(5)
    b = rand(2, 3)
    @test grad(x -> sum(transpose(x)), a)[2][2] == ones(size(a))
    @test grad(x -> sum(adjoint(x)), a)[2][2] == ones(size(a))
    @test grad(x -> sum(transpose(x)), b)[2][2] == ones(size(b))
    @test grad(x -> sum(adjoint(x)), b)[2][2] == ones(size(b))
end


mutable struct Linear{T}
    W::AbstractArray{T,2}
    b::AbstractArray{T}
end

forward(m::Linear, X) = m.W * X .+ m.b

loss(m::Linear, X) = sum(forward(m, X))


mutable struct Point x; y end
constructor_loss(a) = (p = Point(a, a); p.x + p.y)


@testset "grad: structs" begin
    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]
    tape = gradtape(loss, args...)

    @test tape[tape.result].val[1] == loss(args...)

    play!(tape, loss, args...)
    @test tape[tape.result].val[1] == loss(args...)

    val, g = grad(loss, args...)
    @test val == loss(args...)

    _, g = grad(constructor_loss, 4.0)
    @test g[2][1] == 2
end


mutable struct Line p1; p2 end

function add_points(x)
    p = Point(x, x)
    l = Line(p, p)
    return 2*l.p1.x + 5*l.p2.y
end

@testset "grad: structs/new" begin
    _, tape = trace(add_points, rand())

    @test grad(add_points, rand())[2][2] == 7
end


make_t(x) = tuple(x, x)

function make_t_loss(x)
    a, b = make_t(x)
    return a + b
end


@testset "grad: tuple unpack" begin
    _, g = grad(make_t_loss, 3.0)
    @test g[2] == 2.0
end

@testset "grad: tuple index" begin
    y, (_, g) = Yota.grad(x -> x[1].a, ((a=13,), (b=17,)))
    @test y == 13
    @test g isa Tangent{<:Tuple}
    @test g[1] isa Tangent{<:NamedTuple}
    @test g[1].a == 1
    @test g[2] isa NoTangent
end

@testset "grad: seed" begin
    val, g = grad(x -> 2x, [1.0, 2.0, 3.0]; seed=ones(3))
    @test val == [2.0, 4.0, 6.0]
    @test g == (ZeroTangent(), [2.0, 2.0, 2.0])
end


@testset "grad: compiled" begin
    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]
    tape = gradtape(loss, args...)

    # generate new args
    args = Any[Linear(rand(3,4), rand(3)), rand(4,5)]

    # compiled vs. non-compiled
    res1 = play!(tape, loss, args...)
    grad_loss = compile(tape)
    res2 = grad_loss(loss, args...)
    @test res1 == res2

    # fresh vs. cached version
    res3 = grad(loss, args...)
    res4 = grad(loss, args...)
    @test res3 == res2
    @test res4 == res2
end
