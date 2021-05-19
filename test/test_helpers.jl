import Yota: ungetindex, ungetindex!


@testset "helpers: getindex" begin
    x = rand(4)
    dx = zero(x)
    dy = 1.0
    @test ungetindex!(dx, x, dy, 2) == [0.0, 1, 0, 0]
    @test ungetindex(x, dy, 2) == [0.0, 1, 0, 0]

    x = rand(4)
    dx = zero(x)
    dy = [1, 1, 1]
    @test ungetindex!(dx, x, dy, [1, 3, 1]) == [2.0, 0, 1, 0]
    @test ungetindex(x, dy, [1, 3, 1]) == [2.0, 0, 1, 0]

    x = rand(4, 5)
    dx = zero(x)
    dy = ones(4, 3)
    expected = [2 0 1 0 0;
                2 0 1 0 0;
                2 0 1 0 0;
                2 0 1 0 0.0]
    @test ungetindex!(dx, x, dy, :, [1, 3, 1]) == expected
    @test ungetindex(x, dy, :, [1, 3, 1]) == expected

    if CUDA.functional()
        x = rand(4) |> cu
        dx = zero(x) |> cu
        dy = 1.0f0
        @test ungetindex!(dx, x, dy, 2) == cu([0.0, 1, 0, 0])
        @test ungetindex(x, dy, 2) == cu([0.0, 1, 0, 0])

        x = rand(4, 5) |> cu
        dx = zero(x) |> cu
        dy = ones(4, 3) |> cu
        @test ungetindex!(dx, x, dy, :, cu([1, 3, 1])) == cu(expected)
        @test ungetindex(x, dy, :, cu([1, 3, 1])) == cu(expected)
    end

end
