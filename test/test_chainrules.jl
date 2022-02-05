import ChainRulesTestUtils.test_rrule
import ChainRulesCore.rrule
import Yota: isprimitive, CR_CTX


double_inc(x) = 2x + 1


primitive_test(x; y=1) = x + y
primitive_test(x, y) = x + y
primitive_test2(x; y=1) = x + y
primitive_test2(x, y) = x + y

# rrule(::typeof(primitive_test), x) = primitive_test(x), dy -> (NoTangent(), 1)
rrule(::typeof(primitive_test), x; y=1) = primitive_test(x; y=y), dy -> (NoTangent(), 1)
# rrule(::YotaRuleConfig, ::typeof(primitive_test2), x) = primitive_test2(x), dy -> (NoTangent(), 1)
rrule(::YotaRuleConfig, ::typeof(primitive_test2), x; y=1) = primitive_test2(x; y=y), dy -> (NoTangent(), 1)


@testset "rrule utils" begin

    rr = make_rrule(double_inc, 2.0)
    val, pb = rr(double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (ZeroTangent(), 2.0)

    config = YotaRuleConfig()
    val, pb = rrule_via_ad(config, double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (ZeroTangent(), 2.0)

    x, y = rand(2)
    @test isprimitive(CR_CTX, primitive_test, x) == true
    @test isprimitive(CR_CTX, Core.kwfunc(primitive_test), (y=1,), primitive_test, x) == true
    @test isprimitive(CR_CTX, primitive_test, x, y) == false

    @test isprimitive(CR_CTX, primitive_test2, x) == true
    @test isprimitive(CR_CTX, Core.kwfunc(primitive_test2), (y=1,), primitive_test, x,) == true
    @test isprimitive(CR_CTX, primitive_test2, x, y) == false
end