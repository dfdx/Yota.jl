import ChainRulesTestUtils.test_rrule
import ChainRulesCore: rrule, unthunk
import Yota: isprimitive, CR_CTX


double_inc(x::Number) = 2x + 1
double_inc(x::AbstractArray) = 2x .+ 1


primitive_test(x; y=1) = x + y
primitive_test(x, y) = x + y
primitive_test2(x; y=1) = x + y
primitive_test2(x, y) = x + y

rrule(::typeof(primitive_test), x; y=1) = primitive_test(x; y=y), dy -> (NoTangent(), 1)
rrule(::YotaRuleConfig, ::typeof(primitive_test2), x; y=1) = primitive_test2(x; y=y), dy -> (NoTangent(), 1)


@testset "chainrules api" begin

    rr = make_rrule(double_inc, 2.0)
    val, pb = rr(double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (NoTangent(), 2.0)

    config = YotaRuleConfig()
    val, pb = rrule_via_ad(config, double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (NoTangent(), 2.0)

    x = rand(3)
    val, pb = rrule_via_ad(config, double_inc, x)
    @test val == double_inc(x)
    dxs = map(unthunk, pb(ones(3)))
    @test dxs == (NoTangent(), [2.0, 2.0, 2.0])
    dxs = map(unthunk, pb([1, 2, 3]))
    @test dxs == (NoTangent(), [2.0, 4.0, 6.0])

    x, y = rand(2)
    @test isprimitive(CR_CTX, primitive_test, x) == true
    @test isprimitive(CR_CTX, Core.kwfunc(primitive_test), (y=1,), primitive_test, x) == true
    @test isprimitive(CR_CTX, primitive_test, x, y) == false

    @test isprimitive(CR_CTX, primitive_test2, x) == true
    @test isprimitive(CR_CTX, Core.kwfunc(primitive_test2), (y=1,), primitive_test, x,) == true
    @test isprimitive(CR_CTX, primitive_test2, x, y) == false
end