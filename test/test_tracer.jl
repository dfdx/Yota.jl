inc_mul(a::Real, b::Real) = a * (b + 1.0)
inc_mul(A::AbstractArray, B::AbstractArray) = inc_mul.(A, B)
inc_mul2(A::AbstractArray, B::AbstractArray) = A .* (B .+ 1)


@testset "tracer: calls" begin
    val, tape = trace(inc_mul, 2.0, 3.0)
    @test val == inc_mul(2.0, 3.0)
    @test length(tape) == 5
    @test tape[3] isa Constant
end

@testset "tracer: bcast" begin
    A = rand(3)
    B = rand(3)
    val, tape = trace(inc_mul, A, B)
    @test val == inc_mul(A, B)
    # broadcasting may be lowered to different forms,
    # so making no assumptions regarfing the tape

    val, tape = trace(inc_mul2, A, B)
    @test val == inc_mul2(A, B)
end
