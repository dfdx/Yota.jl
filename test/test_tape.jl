@testset "tape: record" begin

    tape = Tape()
    xv = rand(3,4)
    yv = rand(4,3)
    x = record!(tape, Input, xv)
    y = record!(tape, Input, yv)
    u = record!(tape, Call, *, (x, y))
    c = record!(tape, Constant, 2.0)
    w = record!(tape, Bcast, *, (u, c))
    z = record!(tape, Call, sum, (u,); kwargs=Dict{Symbol,Any}(:dims=>1))

    uv = xv * yv
    wv = uv .* 2
    zv = sum(uv, dims=1)
    @test getvalue(z) == zv

end


@testset "tape: ops" begin

    tape = Tape()
    xv = rand(3,4)
    yv = rand(4,3)
    x = record!(tape, Input, xv)
    y = record!(tape, Input, yv)
    z = sum(x * y .* 2; dims=1)

    zv = sum(xv * yv .* 2; dims=1)
    @test getvalue(z) == zv

end
