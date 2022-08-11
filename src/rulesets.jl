import ChainRulesCore: rrule
import Umlaut: __new__

###############################################################################
#                                 Rules                                       #
###############################################################################


# function rrule(::YotaRuleConfig, ::typeof(Core._apply_iterate),
#         ::typeof(iterate), f::F, args...) where F
#     # flatten nested arguments
#     flat = []
#     for a in args
#         push!(flat, a...)
#     end
#     # apply rrule of the function on the flat arguments
#     y, pb = rrule_via_ad(YotaRuleConfig(), f, flat...)
#     sizes = map(length, args)
#     function _apply_iterate_pullback(dy)
#         if dy isa NoTangent
#             return ntuple(_-> NoTangent(), length(args) + 3)
#         end
#         flat_dargs = pb(dy)
#         df = flat_dargs[1]
#         # group derivatives to tuples of the same sizes as arguments
#         dargs = []
#         j = 2
#         for i = 1:length(args)
#             darg_val = flat_dargs[j:j + sizes[i] - 1]
#             if length(darg_val) == 1 && darg_val[1] isa NoTangent
#                 push!(dargs, darg_val[1])
#             else
#                 darg = Tangent{typeof(darg_val)}(darg_val...)
#                 push!(dargs, darg)
#             end
#             j = j + sizes[i]
#         end
#         return NoTangent(), NoTangent(), df, dargs...
#     end
#     return y, _apply_iterate_pullback
# end


function ChainRulesCore.rrule(::YotaRuleConfig, ::typeof(tuple), args...)
    y = tuple(args...)
    N = length(args)
    function tuple_pullback(Δ)
        Δ = unthunk(Δ)
        δargs = (Δ isa NoTangent || Δ isa ZeroTangent) ? [Δ for _=1:N] : collect(Δ)
        (NoTangent(), δargs...)
    end
    return y, tuple_pullback
end


function ChainRulesCore.rrule(::YotaRuleConfig, nt::Type{NamedTuple{names}}, t::Tuple) where {names}
    val = nt(t)
    function namedtuple_pullback(dy)
        return NoTangent(), dy
    end
    return val, namedtuple_pullback
end


# test_rrule(YotaRuleConfig(), NamedTuple{(:dims,)}, (1,))


ChainRulesCore.rrule(::Type{Val{V}}) where V = Val{V}(), dy -> (NoTangent(),)


###############################################################################
#                                 Broadcast                                   #
###############################################################################

# rrules for broadcasting are already defined in ChainRules
# see for details: https://github.com/JuliaDiff/ChainRules.jl/pull/644
# yet we need an rrule for materialize() and a few utilities for rrule_via_ad

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


function ChainRulesCore.rrule(::YotaRuleConfig, ::typeof(Broadcast.materialize), x)
    return Broadcast.materialize(x), dy -> (NoTangent(), dy)
end



# # test_rrule(Broadcast.materialize, Broadcast.broadcasted(sin, rand(3)); output_tangent=ones(3))


# function rrule(::YotaRuleConfig, ::typeof(Broadcast.broadcasted), ::typeof(+), x, y)
#     function plus_bcast_pullback(dy)
#         # println("dy- $dy")
#         dy = unthunk(dy)
#         # println("dy+ $dy")
#         return NoTangent(), NoTangent(), unbroadcast(x, dy), unbroadcast(y, dy)
#     end
#     return x .+ y, plus_bcast_pullback
# end


# function rrule(::YotaRuleConfig, ::typeof(Broadcast.broadcasted), ::typeof(*), x, y)
#     function mul_bcast_pullback(dy)
#         dy = unthunk(dy)
#         return NoTangent(), NoTangent(), unbroadcast_prod_x(x, y, dy), unbroadcast_prod_y(x, y, dy)
#     end
#     return x .* y, mul_bcast_pullback
# end


# function rrule(::YotaRuleConfig, ::typeof(Broadcast.broadcasted), ::typeof(Base.literal_pow),
#         ::typeof(^), x, ::Val{p}) where p
#     y = Broadcast.broadcasted(Base.literal_pow, ^, x, Val(p))
#     function literal_pow_pullback(dy)
#         dy = unthunk(dy)
#         return NoTangent(), NoTangent(), NoTangent(), (@. p * x ^ (p - 1) * dy), ZeroTangent()
#     end
#     return y, literal_pow_pullback
# end


# function rrule(::YotaRuleConfig, ::typeof(Broadcast.broadcasted), ::typeof(^), x, p::Real)
#     y = x .^ p
#     function pow_pullback(dy)
#         dy = unthunk(dy)
#         return NoTangent(), NoTangent(), (@. p * x ^ (p - 1) * dy), ZeroTangent()
#     end
#     return y, pow_pullback
# end


###############################################################################
#                        getindex, getfield, __new__                          #
###############################################################################

function rrule(::YotaRuleConfig, ::typeof(getindex), x, I...)
    y = getindex(x, I...)
    function getindex_pullback(dy)
        dy = unthunk(dy)
        return NoTangent(), ungetindex(x, dy, I...), [ZeroTangent() for i in I]...
    end
    return y, getindex_pullback

end


function rrule(::YotaRuleConfig, ::typeof(getproperty), s, f::Symbol)
    y = getproperty(s, f)
    function getproperty_pullback(dy)
        dy = unthunk(dy)
        T = typeof(s)
        nt = NamedTuple{(f,)}((dy,))
        return NoTangent(), Tangent{T}(; nt...), ZeroTangent()
    end
    return y, getproperty_pullback
end


function rrule(::YotaRuleConfig, ::typeof(getfield), s::Tuple, f::Int)
    y = getfield(s, f)
    function tuple_getfield_pullback(dy)
        dy = unthunk(dy)
        T = typeof(s)
        # deriv of a tuple is a Tangent{Tuple}(...) with all elements set to ZeroTangent()
        # except for the one at index f which is set to dy
        return NoTangent(), Tangent{T}([i == f ? dy : ZeroTangent() for i=1:length(s)]...), ZeroTangent()
    end
    return y, tuple_getfield_pullback
end


function rrule(::YotaRuleConfig, ::typeof(__new__), T, args...)
    y = __new__(T, args...)
    function __new__pullback(dy)
        dy = unthunk(dy)
        if dy isa NoTangent || dy isa ZeroTangent
            fld_derivs = [dy for fld in fieldnames(T)]
        else
            fld_derivs = [getproperty(dy, fld) for fld in fieldnames(T)]
        end
        return NoTangent(), NoTangent(), fld_derivs...
    end
    return y, __new__pullback
end


###############################################################################
#                                   iterate                                   #
###############################################################################

function rrule(::YotaRuleConfig, ::typeof(iterate), x::AbstractArray)
    y = iterate(x)
    function iterate_pullback(dy)
        dy = unthunk(dy)
        return NoTangent(), ungetindex(x, dy, 1)
    end
    return y, iterate_pullback
end

function rrule(::YotaRuleConfig, ::typeof(iterate), x::AbstractArray, i::Integer)
    y = iterate(x, i)
    function iterate_pullback(dy)
        dy = unthunk(dy)
        return NoTangent(), ungetindex(x, dy, i), ZeroTangent()
    end
    return y, iterate_pullback
end

function rrule(::YotaRuleConfig, ::typeof(iterate), t::Tuple)
    y = iterate(t)
    function iterate_pullback(dy)
        dy = unthunk(dy)
        return NoTangent(), ungetfield(dy[1], t, 1)
    end
    return y, iterate_pullback
end

function rrule(::YotaRuleConfig, ::typeof(iterate), t::Tuple, i::Integer)
    y = iterate(t, i)
    function iterate_pullback(dy)
        dy = unthunk(dy)
        return NoTangent(), ungetfield(dy[1], t, i), ZeroTangent()
    end
    return y, iterate_pullback
end


# here we explicitely stop propagation in iteration
# over ranges (e.g for i=1:3 ... end)
function rrule(::YotaRuleConfig, ::typeof(iterate), x::UnitRange)
    y = iterate(x)
    function iterate_pullback(dy)
        return NoTangent(), ZeroTangent()
    end
    return y, iterate_pullback
end
function rrule(::YotaRuleConfig, ::typeof(iterate), x::UnitRange, i::Integer)
    y = iterate(x, i)
    function iterate_pullback(dy)
        return NoTangent(), ZeroTangent(), ZeroTangent()
    end
    return y, iterate_pullback
end


## tuple unpacking

function rrule(::YotaRuleConfig, ::typeof(Base.indexed_iterate), t::Tuple, i::Int)
    y = Base.indexed_iterate(t, i)
    function indexed_iterate_pullback(dy)
        d_val, d_state = dy
        return NoTangent(), ungetfield(d_val, t, i), ZeroTangent()
    end
    return y, indexed_iterate_pullback
end

function rrule(::YotaRuleConfig, ::typeof(Base.indexed_iterate), t::Tuple, i::Int, state::Int)
    y = Base.indexed_iterate(t, i, state)
    function indexed_iterate_pullback(dy)
        d_val, d_state = dy
        return NoTangent(), ungetfield(d_val, t, i), ZeroTangent(), ZeroTangent()
    end
    return y, indexed_iterate_pullback
end

function rrule(::YotaRuleConfig, ::Colon, a::Int, b::Int)
    y = a:b
    function colon_pullback(dy)
        return NoTangent(), NoTangent(), NoTangent()
    end
    return y, colon_pullback
end