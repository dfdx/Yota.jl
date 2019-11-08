@testset "helpers" begin
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
