
struct DeviceTestStruct
    a
    b
end

@testset "devices" begin
    if CUDA.functional()
        cpu = CPU()
        gpu = GPU()

        A = rand(2, 3)
        cA = cu(A)
        @test gpu(A) == cA
        @test cpu(cA) ≈ A

        @test all(gpu((A, 1.0)) .== (cA, 1f0))
        @test all(cpu((cA, 1f0)) .≈ (A, 1f0))

        s = DeviceTestStruct(A, 1.0)
        cs = gpu(s)
        @test cs.a == cu(s.a)
        @test cs.b == 1f0 && typeof(cs.b) == Float32
        @test cpu(cs).a ≈ A
        @test cpu(cs).b == 1f0
    end
end