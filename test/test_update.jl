import ChainRulesCore.Tangent
import Yota: update!, setfield_nested!


mutable struct StructA
    t::Array{Float64, 1}
    s::Float64
end

mutable struct StructB
    a::StructA
    s::Float64
end


@testset "update" begin

    # setfield_nested
    b = StructB(StructA([1, 2, 3], 4), 5)
    setfield_nested!(b, (:a, :t), [-1.0, -2.0, -3.0])
    setfield_nested!(b, (:a, :s), -4.0)
    setfield_nested!(b, (:s,), -5.0)
    @test b.a.t == [-1.0, -2.0, -3.0]
    @test b.a.s == -4.0
    @test b.s == -5.0

    # struct update
    g = Dict((:a, :t) => [1.0, 2.0, 3.0],
             (:a, :s) => 4.0,
             (:s,) => 5.0)
    g = Tangent{StructB}(s=5.0, a=Tangent{StructA}(t=[1.0, 2.0, 3.0], s=4.0))
    update!(b, g)
    @test b.a.t == [-2.0, -4.0, -6.0]
    @test b.a.s == -8.0
    @test b.s == -10.0

    # struct update with function
    update!(b, g, (x, g) -> x .+ 2g)
    @test b.a.t == [0, 0, 0]
    @test b.a.s == 0
    @test b.s == 0

    # array update
    x = rand(3,4)
    update!(x, x)
    @test x == zero(x)

    # array update with function
    x = rand(3,4)
    xo = copy(x)
    update!(x, x, (x, g) -> x - 2g)
    @test x == -xo

    # with actual grad update
    b = StructB(StructA(ones(4), 1.0), 1.0)
    _, g = grad(b -> sum(b.a.t) + (b.a.s + b.s), b)
    update!(b, g[2], (x, gx) -> x .- 0.5gx)
    @test b.a.t == [0.5, 0.5, 0.5, 0.5]
    @test b.a.s == 0.5
    @test b.s == 0.5
end
