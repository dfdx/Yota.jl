non_binarized(a, b) = sqrt(a * b * b) + a + a
complex_non_binarized(x) = (x[3]*x[8]^3 - 3*x[3]*x[8]*x[6]^2 - x[1]*x[6]^3 +
                            3*x[1]*x[6]*x[8]^2 + x[4]*x[5]^3 - 3*x[4]*x[5]*x[7]^2 -
                            x[2]*x[7]^3 + 3*x[2]*x[7]*x[5]^2 - 1.2342523)

function func_with_loop(x)
    for i=1:5
        x = 2 .* x
    end
    return sum(x)
end


@testset "Tape utils" begin
    _, tape = trace(non_binarized, 4.0, 2.0)
    b_tape = binarize_ops(tape)
    @test play!(tape, 1.0, 2.0) == non_binarized(1.0, 2.0)
    @test play!(tape, 1.0, 2.0) == play!(b_tape, 1.0, 2.0)

    x = rand(8)
    _, tape = trace(complex_non_binarized, x)
    b_tape = binarize_ops(tape)
    x2 = rand(8)
    @test play!(tape, x2) == complex_non_binarized(x2)
    @test play!(tape, x2) == play!(b_tape, x2)

    _, tape = trace(func_with_loop, x)
    reduced_tape = remove_unused(tape)
    @test length(reduced_tape) = 17   # may be too strict, but let's give it a try
    @test play!(reduced_tape, x) == play!(tape, x)

    # test unwind_iterate
    _, tape = trace(t -> (r = 0; for x in t r += x end; return r), (100, 200, 300); optimize=false)
    tape_no_iter = remove_unused(unwind_iterate(tape))
    @test play!(tape, (1, 2, 3)) == play!(tape, (1, 2, 3))
    @test length(tape_no_iter) < length(tape)


    tape = Tape()
    record!(tape, Input, 3.0)
    record!(tape, Input, 2.0)
    record!(tape, Call, 5.0, +, [1, 2])
    record!(tape, Constant, 4.0)
    record!(tape, Call, 5.0, +, [1, 2])
    record!(tape, Call, 25.0, *, [3, 5])
    tape.resultid = length(tape)

    tape2 = eliminate_common(tape)
    @test length(tape2) == 5
    @test play!(tape, 9.0, 4.0) == play!(tape2, 9.0, 4.0)

end
