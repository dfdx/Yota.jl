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
end
