@testset "grad: simple" begin
    args3 = (rand(4, 3), rand(4), rand(3))
    args4 = (rand(4, 3), rand(4), rand(3), rand(4))
    @test gradcheck((W, b, x) -> sum(W * x .+ b), args3...)
    @test gradcheck((W, b, x) -> sum(tanh.(W * x .+ b)), args3...)
    @test gradcheck((W, b, x, y) -> sum(abs2.(tanh.(W * x .+ b) .- y)), args4...)
end
