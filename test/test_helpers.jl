@testset "helpers: cat" begin
    # unvcat: 1D
    arrs = Any[i * ones(i) for i=1:4]
    A = 2 * vcat(arrs...)
    for i=1:4
        @test 2 * arrs[i] == unvcat(A, i, arrs...)
    end

    # unvcat: 2D
    arrs = Any[i * ones(i, 3) for i=1:4]
    A = 2 * vcat(arrs...)
    for i=1:4
        @test 2 * arrs[i] == unvcat(A, i, arrs...)
    end

    # unvcat: 4D
    arrs = Any[i * ones(i, 3, 4, 5) for i=1:4]
    A = 2 * vcat(arrs...)
    for i=1:4
        @test 2 * arrs[i] == unvcat(A, i, arrs...)
    end

    # unvcat: 2D
    arrs = Any[i * ones(3, i) for i=1:4]
    A = 2 * hcat(arrs...)
    for i=1:4
        @test 2 * arrs[i] == unhcat(A, i, arrs...)
    end

    # unvcat: 4D
    arrs = Any[i * ones(3, i, 5, 6) for i=1:4]
    A = 2 * hcat(arrs...)
    for i=1:4
        @test 2 * arrs[i] == unhcat(A, i, arrs...)
    end

    # uncat: 4D
    arrs = Any[i * ones(3, 4, i, 6) for i=1:4]
    A = 2 * cat(arrs...; dims=3)
    for i=1:4
        @test 2 * arrs[i] == uncat(A, i, arrs...; dims=3)
    end

    @test gradcheck((a, b, c, d) -> (sum(cat(a, b, c, d; dims=3))), arrs...)
end


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
