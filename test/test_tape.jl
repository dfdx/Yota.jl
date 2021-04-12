import Yota: Tape, V, inputs!, rebind!, mkcall


@testset "tape" begin
    # rebind!
    tape = Tape()
    v1, v2 = inputs!(tape, nothing, 3.0, 5.0)
    v3 = push!(tape, mkcall(*, v1, 2))
    st = Dict(v1.id => v2.id)
    rebind!(tape, st)
    @assert tape[v3].args[1].id == v2.id


    # push!, insert!, replace!
    tape = Tape()
    a1, a2, a3 = inputs!(tape, nothing, 2.0, 5.0)
    r = push!(tape, mkcall(*, a2, a3))
    @test tape[r].val == 10.0

    ops = [mkcall(+, a2, 1), mkcall(+, a3, 1)]
    v1, v2 = insert!(tape, 4, ops...)
    @test r.id == 6

    tape[r] = mkcall(*, v1, v2)
    @test tape[r].val == 18.0

    v2.id = 100
    @test tape[r].args[2].id == 100

    op1 = mkcall(*, V(2), 2)
    op2 = mkcall(+, V(op1), 1)
    replace!(tape, 4 => [op1, op2]; rebind_to=2)
    @test tape[V(7)].args[1].id == op2.id
end