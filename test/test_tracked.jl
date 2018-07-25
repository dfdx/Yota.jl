@testset "tracked: accessors" begin

    tape = Tape()
    r = rand()
    tr = tracked(tape, r)

    @test gettape(tr) == tape
    @test getid(tr) == -1
    @test getvalue(tr) == r
    setvalue!(tr, r + 1.0)
    @test getvalue(tr) == r + 1.0

    a = rand(3,4)
    ta = tracked(tape, a)

    @test gettape(ta) == tape
    @test getid(ta) == -1
    @test getvalue(ta) == a
    setvalue!(ta, a .+ 1)
    @test getvalue(ta) == a .+ 1

end


@testset "tracked: overloaded ops" begin

    tape = Tape()
    xv = rand(); yv = rand()
    x = tracked(tape, xv)
    y = tracked(tape, yv)
    z = 2x^2 + y
    @test getvalue(z) == 2xv^2 + yv
    @test length(tape) == 5

    tape = Tape()
    xv = rand(3,4); yv = rand(4,3)    
    x = tracked(tape, xv); y = tracked(tape, yv)
    z = x * y .+ 1.0
    @test getvalue(z) == xv * yv .+ 1.0
    @test length(tape) == 3

end
