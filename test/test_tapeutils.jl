non_binarized(a, b) = sqrt(a * b * b) + a + a
complex_non_binarized(x) = (x[3]*x[8]^3 - 3*x[3]*x[8]*x[6]^2 - x[1]*x[6]^3 +
                            3*x[1]*x[6]*x[8]^2 + x[4]*x[5]^3 - 3*x[4]*x[5]*x[7]^2 -
                            x[2]*x[7]^3 + 3*x[2]*x[7]*x[5]^2 - 1.2342523)


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
end
