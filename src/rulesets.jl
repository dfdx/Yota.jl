import ChainRulesCore: rrule

###############################################################################
#                                 Rules                                       #
###############################################################################


# TODO: figure out failing test_rrules, move to actual tests


# function rrule(::typeof(Broadcast.broadcasted), ::typeof(identity), x)
#     identity_pullback(dy) = (NoTangent(), NoTangent(), dy)
#     return x, identity_pullback
# end

# test_rrule(broadcasted, identity, [1.0, 2.0]) -- fails at the moment


function rrule(::YotaRuleConfig, ::typeof(Core._apply_iterate),
        ::typeof(iterate), f::F, args...) where F
    # flatten nested arguments
    flat = []
    for a in args
        push!(flat, a...)
    end
    # apply rrule of the function on the flat arguments
    y, pb = rrule_via_ad(YotaRuleConfig(), f, flat...)
    sizes = map(length, args)
    function _apply_iterate_pullback(dy)
        if dy isa NoTangent
            return ntuple(_-> NoTangent(), length(args) + 3)
        end
        flat_dargs = pb(dy)
        df = flat_dargs[1]
        # group derivatives to tuples of the same sizes as arguments
        dargs = []
        j = 2
        for i = 1:length(args)
            darg_val = flat_dargs[j:j + sizes[i] - 1]
            if length(darg_val) == 1 && darg_val[1] isa NoTangent
                push!(dargs, darg_val[1])
            else
                darg = Tangent{typeof(darg_val)}(darg_val...)
                push!(dargs, darg)
            end
            j = j + sizes[i]
        end
        return NoTangent(), NoTangent(), df, dargs...
    end
    return y, _apply_iterate_pullback
end


function ChainRulesCore.rrule(::typeof(tuple), args...)
    y = tuple(args...)
    return y, dy -> (NoTangent(), collect(dy)...)
end

# test_rrule(tuple, 1, 2, 3; output_tangent=Tangent{Tuple}((1, 2, 3)), check_inferred=false)


function rrule(nt::Type{NamedTuple{names}}, t::Tuple) where {names}
    val = nt(t)
    function namedtuple_pullback(dy)
        return NoTangent(), dy
    end
    return val, namedtuple_pullback
end


# test_rrule(YotaRuleConfig(), NamedTuple{(:dims,)}, (1,))


rrule(::Type{Val{V}}) where V = Val{V}(), dy -> (NoTangent(),)


###############################################################################
#                                 Broadcast                                   #
###############################################################################

# unzip taken from Zygote:
# https://github.com/FluxML/Zygote.jl/blob/d5be4d5ca80e79278d714eaac15ca71904a262e3/src/lib/array.jl#L177-L185
struct StaticGetter{i} end
(::StaticGetter{i})(v) where {i} = v[i]

@generated function _unzip(tuples, ::Val{N}) where {N}
  Expr(:tuple, (:(map($(StaticGetter{i}()), tuples)) for i ∈ 1:N)...)
end

function unzip(tuples)
  N = length(first(tuples))
  _unzip(tuples, Val(N))
end


function rrule(::typeof(Broadcast.broadcasted), f::F, args...) where F
    ys, pbs = unzip(rrule_via_ad.(YOTA_RULE_CONFIG, f, args...))
    function pullback(Δ)
        Δ = unthunk(Δ)
        dxs = map((pb, Δ) -> pb(Δ), pbs, Δ) |> unzip
        dxs = [all(dx .== NoTangent()) ? NoTangent() : dx for dx in dxs]
        return NoTangent(), dxs...
    end
    return ys, pullback
end


function rrule(::typeof(Broadcast.materialize), x)
    return Broadcast.materialize(x), dy -> (NoTangent(), dy)
end

# test_rrule(Broadcast.materialize, Broadcast.broadcasted(sin, rand(3)); output_tangent=ones(3))


# TODO: turn into rrules


# function ∇broadcasted_special(dy, ::typeof(broadcasted), ::typeof(+), x, y)
#     return NoTangent(), NoTangent(), unbroadcast(x, dy), unbroadcast(y, dy)
# end
# @drule broadcasted(::typeof(+), x::Any, y::Any) ∇broadcasted_special

# function ∇broadcasted_special(dy, ::typeof(broadcasted), ::typeof(*), x, y)
#     return NoTangent(), NoTangent(), unbroadcast_prod_x(x, y, dy), unbroadcast_prod_y(x, y, dy)
# end
# @drule broadcasted(::typeof(*), x::Any, y::Any) ∇broadcasted_special

# function ∇broadcasted(dy, ::typeof(broadcasted), ::typeof(Base.literal_pow),
#     ::typeof(^), x::Any, ::Val{p}) where p
#     return NoTangent(), NoTangent(), NoTangent(), (@. p * x ^ (p - 1) * dy), ZeroTangent()
# end
# @drule broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Any, ::Val) ∇broadcasted

# function ∇broadcasted(dy, ::typeof(broadcasted),
#     ::typeof(^), x::Any, p::Real)
#     return NoTangent(), NoTangent(), (@. p * x ^ (p - 1) * dy), ZeroTangent()
# end
# @drule broadcasted(::typeof(^), x::Any, ::Real) ∇broadcasted