import Yota: trace, V, Call

inc_mul(a::Real, b::Real) = a * (b + 1.0)
inc_mul(A::AbstractArray, B::AbstractArray) = inc_mul.(A, B)
inc_mul2(A::AbstractArray, B::AbstractArray) = A .* (B .+ 1)

non_primitive(x) = 2x + 1
non_primitive_caller(x) = sin(non_primitive(x))


@testset "trace" begin
    # calls
    val, tape = trace(inc_mul, 2.0, 3.0)
    @test val == inc_mul(2.0, 3.0)
    @test length(tape) == 5
    @test tape[V(5)].args[1].id == 2

    # bcast
    A = rand(3)
    B = rand(3)
    val, tape = trace(inc_mul, A, B)
    @test val == inc_mul(A, B)
    # broadcasting may be lowered to different forms,
    # so making no assumptions regarding the tape

    val, tape = trace(inc_mul2, A, B)
    @test val == inc_mul2(A, B)

    # primitives
    x = 3.0
    val1, tape1 = trace(non_primitive_caller, x)
    val2, tape2 = trace(non_primitive_caller, x; primitives=Set([non_primitive, sin]))

    @test val1 == val2
    @test any(op isa Call && op.fn == (*) for op in tape1)
    @test tape2[V(3)].fn == non_primitive
    @test tape2[V(4)].fn == sin
end

