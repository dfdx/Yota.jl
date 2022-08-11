import ChainRulesCore.rrule
import Yota.YotaRuleConfig


@testset "tuple" begin
    test_rrule(YotaRuleConfig(), tuple, 1.0, 2.0, 3.0; check_inferred=false)
end


# broacastable non-primitive
sin_inc(x::Number) = sin(x) + 1

@testset "generic broadcasted" begin
    # numeric gradient in test_rrule() gives highly inaccurate results
    # thus using gradcheck instead
    # test_rrule(YotaRuleConfig(), Broadcast.materialize, Broadcast.broadcasted(sin, rand(3)))

    @test gradcheck(x -> sum(sin.(x)), rand(3))

    for f in [sin, sin_inc]
        xs = rand(2)

        # manually get pullbacks for each element and apply them to seed 1.0
        pbs = !isnothing(rrule(f, xs[1])) ?
                [rrule(f, x)[2] for x in xs] :       # just in case
                [rrule_via_ad(YotaRuleConfig(), f, x)[2] for x in xs]
        dxs = [pbs[1](1.0)[2], pbs[2](1.0)[2]]

        # use rrule for broadcasted
        style = Broadcast.combine_styles(xs)
        _, bcast_pb = rrule(YotaRuleConfig(), Broadcast.broadcasted, style, f, xs)
        dxs_bcast = bcast_pb(ones(2))[end]

        @test all(dxs .== dxs_bcast)
    end
end