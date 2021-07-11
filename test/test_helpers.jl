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

    ## additional smoke tests

    x = rand(3, 4)
    I = (1:2, 2:3)
    y = x[I...]
    dy = ones(size(y))
    dx = zero(x)
    ungetindex!(dx, x, dy, I...)

    x = rand(3, 4)
    I = (1, 2:3)
    y = x[I...]
    dy = ones(size(y))
    dx = zero(x)
    ungetindex!(dx, x, dy, I...)

    x = rand(3, 4)
    I = (1, :)
    y = x[I...]
    dy = ones(size(y))
    dx = zero(x)
    ungetindex!(dx, x, dy, I...)

    x = rand(3, 4)
    I = (1, [1, 3])
    y = x[I...]
    dy = ones(size(y))
    dx = zero(x)
    ungetindex!(dx, x, dy, I...)

    # single-element Cartesian index
    x = rand(3, 4)
    I = (1, 2)
    y = x[I...]
    dy = 1.0
    dx = zero(x)
    ungetindex!(dx, x, dy, I...)

    # single-element linear index
    x = rand(3, 4)
    I = (5,)
    y = x[I...]
    dy = 1.0
    dx = zero(x)
    ungetindex!(dx, x, dy, I...)

    # CUDA
    if CUDA.functional()
        x = rand(4) |> cu
        dx = zero(x) |> cu
        dy = 1.0f0
        @test ungetindex!(dx, x, dy, 2) == cu([0.0, 1, 0, 0])
        @test ungetindex(x, dy, 2) == cu([0.0, 1, 0, 0])

        x = rand(4, 5) |> cu
        dx = zero(x) |> cu
        dy = ones(4, 3) |> cu
        @test ungetindex!(dx, x, dy, :, [1, 3, 1]) == cu(expected)
        @test ungetindex(x, dy, :, [1, 3, 1]) == cu(expected)
    end

end