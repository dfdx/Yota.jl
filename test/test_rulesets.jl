import ChainRulesCore.rrule
import Yota.YotaRuleConfig
import Yota.Umlaut.__new__


# broacastable non-primitive
sin_inc(x::Number) = sin(x) + 1
struct PointRS x; y end
config = YotaRuleConfig()


@testset "rulesets" begin

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
                    [rrule_via_ad(config, f, x)[2] for x in xs]
            dxs = [pbs[1](1.0)[2], pbs[2](1.0)[2]]

            # use rrule for broadcasted
            style = Broadcast.combine_styles(xs)
            _, bcast_pb = rrule(config, Broadcast.broadcasted, style, f, xs)
            dxs_bcast = bcast_pb(ones(2))[end]

            @test all(dxs .== dxs_bcast)
        end

        test_rrule(config, Broadcast.materialize, broadcasted(identity, [1.0, 2.0]))

        for arg in ([1.0, 2.0], broadcasted(identity, [1.0, 2.0]), 1.0)
            test_rrule(config, Broadcast.broadcastable, arg)
        end
    end

    @testset "getfield & co." begin
        p = PointRS(42.0, 54.0)
        test_rrule(config, getproperty, p, :x; check_inferred=false)
        test_rrule(config, getfield, p, :x; check_inferred=false)

        t = (2.0, 4.0, 8.0)
        test_rrule(config, getfield, t, 3; check_inferred=false)

        test_rrule(config, __new__, PointRS, 42.0, 54.0; check_inferred=false)
    end

    @testset "iterate" begin
        # CRTU fails here due to internal errors, thus using manual tests instead
        x = rand(3)
        val, pb = rrule(config, iterate, x)
        @test pb((1, NoTangent())) == (NoTangent(), [1.0, 0, 0])
        val, pb = rrule(config, iterate, x, 2)
        @test pb((1, NoTangent())) == (NoTangent(), [0, 1.0, 0], ZeroTangent())

        x = (2.0, 4.0, 8.0)
        val, pb = rrule(config, iterate, x)
        @test_broken pb((1, NoTangent())) == (NoTangent(), Tangent{typeof(x)}((1, ZeroTangent(), ZeroTangent())))
        val, pb = rrule(config, iterate, x, 2)
        @test_broken pb((1, NoTangent())) == (NoTangent(), (ZeroTangent(), 1, ZeroTangent()), ZeroTangent())
    end

    @testset "tuple" begin
        test_rrule(config, tuple, 1.0, 2.0, 3.0; check_inferred=false)
        test_rrule(config, NamedTuple{(:dims,)}, (1,))
        test_rrule(config, getindex, (a=42.0, b=54.0), :a; check_inferred=false)
    end

    @testset "misc" begin
        test_rrule(config, convert, Float64, 2.0)
        test_rrule(config, Val{Float64})
    end

    @testset "non differentiable" begin
        test_rrule(config, eltype, [2.0, 3.0])
        test_rrule(config, Base.iterate, 1:3)
        test_rrule(config, Base.iterate, 1:3, 1)
    end

end
