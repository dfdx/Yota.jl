function dynamic_loss(x, n_iter)
    for i=1:n_iter
        x = 2 .* x
    end
    return sum(x)
end


function dynamic_loss2(x)
    if sum(x) >= 0
        x = 2 .* x
    else
        x = 3 .* x
    end
    return sum(x)
end


@testset "grad: dynamic" begin
    x = rand(4)
    _, g1 = grad(dynamic_loss, x, 1; dynamic=true)
    _, g2 = grad(dynamic_loss, x, 2; dynamic=true)
    _, g3 = grad(dynamic_loss, x, 3; dynamic=true)
    @test g1[1] == fill(2, 4)
    @test g2[1] == fill(4, 4)
    @test g3[1] == fill(8, 4)

    _, g1 = grad(dynamic_loss2, fill(1.0, 4); dynamic=true)
    _, g2 = grad(dynamic_loss2, fill(-1, 4); dynamic=true)
    @test g1[1] == fill(2, 4)
    @test g2[1] == fill(3, 4)
end
