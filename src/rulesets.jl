import ChainRulesCore: rrule, @non_differentiable
import Umlaut: __new__

###############################################################################
#                                 Rules                                       #
###############################################################################


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


for T in (AbstractArray, Broadcast.Broadcasted, Number)
    function ChainRulesCore.rrule(::typeof(Broadcast.broadcastable), x::T)
        broadcastable_pullback(dy) = (NoTangent(), dy)
        return x, broadcastable_pullback
    end
end



###############################################################################
#                        getindex, getfield, __new__                          #
###############################################################################

function rrule(::YotaRuleConfig, ::typeof(getproperty), x::T, f::Symbol) where T
    y = getproperty(x, f)
    proj = ProjectTo(x)
    # valT = Val(T)  # perhaps more stable inside closure?
    function getproperty_pullback(dy)
        nt = NamedTuple{(f,)}((unthunk(dy),))
        # not really sure whether this ought to unthunk or not, maybe ProjectTo will anyway, in which case best to be explicit?
        return NoTangent(), proj(Tangent{T}(; nt...)), ZeroTangent()
    end
    return y, getproperty_pullback
end


# from https://github.com/FluxML/Optimisers.jl/pull/105#issuecomment-1229243707
function rrule(::typeof(getfield), x::T, f::Symbol) where T
    y = getfield(x, f)
    proj = ProjectTo(x)
    # valT = Val(T)  # perhaps more stable inside closure?
    function getfield_pullback(dy)
        nt = NamedTuple{(f,)}((unthunk(dy),))
        # not really sure whether this ought to unthunk or not, maybe ProjectTo will anyway, in which case best to be explicit?
        return NoTangent(), proj(Tangent{T}(; nt...)), ZeroTangent()
    end
    return y, getfield_pullback
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
        return NoTangent(), ZeroTangent(), fld_derivs...
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
        return NoTangent(), ungetindex(x, dy[1], 1)
    end
    return y, iterate_pullback
end

function rrule(::YotaRuleConfig, ::typeof(iterate), x::AbstractArray, i::Integer)
    y = iterate(x, i)
    function iterate_pullback(dy)
        dy = unthunk(dy)
        return NoTangent(), ungetindex(x, dy[1], i), ZeroTangent()
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


###############################################################################
#                                   convert                                   #
###############################################################################

function rrule(::typeof(convert), ::Type{T}, x::T) where T
    return x, Δ -> (NoTangent(), ZeroTangent(), Δ)
end


###############################################################################
#                             non-differentiable                              #
###############################################################################

@non_differentiable Base.lastindex(::NTuple)
@non_differentiable Base.eltype(::Any)
