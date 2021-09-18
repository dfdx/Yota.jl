import ChainRulesTestUtils.test_rrule


double_inc(x) = 2x + 1

@testset "rrule utils" begin

    rr = make_rrule(double_inc, 2.0)
    val, pb = rr(double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (ZeroTangent(), 2.0)

    config = YotaRuleConfig()
    val, pb = rrule_via_ad(config, double_inc, 3.0)
    @test val == 7
    @test pb(1.0) == (ZeroTangent(), 2.0)
end

@testset "rulesets" begin

    f = reshape
    args = [(rand(3, 4),), (12,)]
    test_rrule(YotaRuleConfig(), Core._apply_iterate, iterate, f, args...; check_inferred=false)

end