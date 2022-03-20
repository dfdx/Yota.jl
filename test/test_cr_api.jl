import ChainRulesTestUtils.test_rrule
import ChainRulesCore: rrule, unthunk
import Yota: isprimitive, CR_CTX

const broadcasted = Broadcast.broadcasted


double_inc(x::Number) = 2x + 1
double_inc(x::AbstractArray) = 2x .+ 1

double_dec(x::Number) = 2x - 1

primitive_test(x; y=1) = x + y
primitive_test(x, y) = x + y
primitive_test2(x; y=1) = x + y
primitive_test2(x, y) = x + y

# rrule(::typeof(primitive_test), x) = primitive_test(x), dy -> (NoTangent(), 1)
rrule(::typeof(primitive_test), x; y=1) = primitive_test(x; y=y), dy -> (NoTangent(), 1)
# rrule(::YotaRuleConfig, ::typeof(primitive_test2), x) = primitive_test2(x), dy -> (NoTangent(), 1)
rrule(::YotaRuleConfig, ::typeof(primitive_test2), x; y=1) = primitive_test2(x; y=y), dy -> (NoTangent(), 1)


@testset "chainrules api" begin
    config = YotaRuleConfig()

    rr = make_rrule(double_inc, 2.0)
    val, pb = rr(config, double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (ZeroTangent(), 2.0)

    rr = make_rrule(broadcasted, double_dec, [1.0, 2.0])
    val, pb = rr(config, broadcasted, double_dec, [3.0, 4.0])
    @test val == [5.0, 7.0]
    @test pb([1, 1]) == (ZeroTangent(), ZeroTangent(), [2.0, 2.0])

    val, pb = rrule_via_ad(config, double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (ZeroTangent(), 2.0)

    x = rand(3)
    val, pb = rrule_via_ad(config, double_inc, x)
    @test val == double_inc(x)
    dxs = map(unthunk, pb(ones(3)))
    @test dxs == (ZeroTangent(), [2.0, 2.0, 2.0])
    dxs = map(unthunk, pb([1, 2, 3]))
    @test dxs == (ZeroTangent(), [2.0, 4.0, 6.0])

    x = rand(3)
    val, pb = rrule_via_ad(config, broadcasted, double_dec, x)
    @test val == broadcast(double_dec, x)
    dxs = map(unthunk, pb(ones(3)))
    @test dxs == (ZeroTangent(), ZeroTangent(), [2.0, 2.0, 2.0])
    dxs = map(unthunk, pb([1, 2, 3]))
    @test dxs == (ZeroTangent(), ZeroTangent(), [2.0, 4.0, 6.0])

    x, y = rand(2)
    @test isprimitive(CR_CTX, primitive_test, x) == true
    @test isprimitive(CR_CTX, Core.kwfunc(primitive_test), (y=1,), primitive_test, x) == true
    @test isprimitive(CR_CTX, primitive_test, x, y) == false

    @test isprimitive(CR_CTX, primitive_test2, x) == true
    @test isprimitive(CR_CTX, Core.kwfunc(primitive_test2), (y=1,), primitive_test, x,) == true
    @test isprimitive(CR_CTX, primitive_test2, x, y) == false
end