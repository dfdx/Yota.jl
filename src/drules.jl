const DRULES = Ref(FunctionResolver{Function}())


function expr2signature(modl::Module, ex::Expr)
    BAD_FORMAT_MSG = "Expected signature as `fn(x::T1, y::T2)`"
    @assert Meta.isexpr(ex, :call) BAD_FORMAT_MSG
    # instead of modl.eval() we could use getfield(modl, :name)
    # but it doesn't work with complex stuff like qualified functions
    # or parametric types
    fn_typ = modl.eval(ex.args[1]) |> typeof
    arg_typs = []
    for x in ex.args[2:end]
        @assert Meta.isexpr(x, :(::)) BAD_FORMAT_MSG
        # Module.eval() is slower, but supports UnionAll
        typ = modl.eval(x.args[end])
        push!(arg_typs, typ)
    end
    return Tuple{fn_typ, arg_typs...}
end


macro drule(ex, df)
    let sig = expr2signature(@__MODULE__, ex)
        quote
            $DRULES[][$sig] = $df
            nothing
        end
    end
end


get_deriv_function(sig) = DRULES[][sig]

is_yota_primitive(sig) = sig in DRULES[]


###############################################################################
#                                    RULES                                    #
###############################################################################

# @diffrule *(u::AbstractArray, v::AbstractArray)    u     dy * transpose(v)

# @diffrule *(u::Real         , v::Real )            v     u * dy
# @diffrule *(u::Real         , v::AbstractArray)    v     u .* dy
# @diffrule *(u::AbstractArray, v::Real )            v     sum(u .* dy)
# @diffrule *(u::AbstractArray, v::AbstractArray)    v     transpose(u) * dy

import Base.Broadcast: broadcasted, materialize


function ∇times(dy, ::typeof(*), x::AbstractArray, y::AbstractArray)
    return NoTangent(), dy * y', x' * dy
end
@drule *(x::AbstractArray, y::AbstractArray) ∇times


function ∇times(dy, ::typeof(*), x::Number, y::Number)
    return NoTangent(), dy * y, dy * x
end
@drule *(x::Number, y::Number) ∇times


function ∇plus(dy, ::typeof(+), x::Number, y::Number)
    return NoTangent(), dy, dy
end
@drule +(x::Number, y::Number) ∇plus


# function ∇broadcasted(dy, ::typeof(Broadcast.broadcasted), f::F, args...) where F
#     df = get_deriv_function(Tuple{F, map(eltype, args)...})
#     # TODO: write a more descriptive error message
#     @assert(df !== nothing,
#         "Broadcasting using Yota rule on a non-primitive function $f")
#     # TODO: use this approach instead:
#     # https://github.com/JuliaDiff/ChainRulesCore.jl/issues/68#issuecomment-834437769
#     darg_tuples = df.(dy, f, args...)
#     darg_single_tuple = tuple(
#         [vcat([t[i] for t in darg_tuples]) for i in 1:length(args) + 1]...
#     )
#     return NoTangent(), darg_single_tuple
# end
# @drule Broadcast.broadcasted(f::Function, args::Vararg) ∇broadcasted

function ∇materialize(dy, ::typeof(Broadcast.materialize), x::Any)
    return NoTangent(), dy
end
@drule Broadcast.materialize(x::Any) ∇materialize


# some special broadcasting rules

function ∇broadcasted(dy, ::typeof(Broadcast.broadcasted), f::typeof(+), args...)
    return NoTangent(), NoTangent(), [dy for a in args]...
end
@drule Broadcast.broadcasted(f::typeof(+), args::Vararg) ∇broadcasted

function ∇broadcasted(dy, ::typeof(Broadcast.broadcasted), f::typeof(*), args...)
    return NoTangent(), NoTangent(), [dy .* a for a in args]...
end
@drule Broadcast.broadcasted(f::typeof(*), args::Vararg) ∇broadcasted

# @diffrule getindex(u::AbstractArray, i)         u    ungetindex(u, dy, i)

function ∇getindex(dy, ::typeof(getindex), x, I...)
    return NoTangent(), ungetindex(x, dy, I...), [ZeroTangent() for i in I]...
end
@drule getindex(x::Any, I::Vararg) ∇getindex


function ∇sum(dy, ::typeof(sum), x::AbstractArray)
    dx = similar(x)
    dx .= dy
    return NoTangent(), dx
end
@drule sum(x::AbstractArray) ∇sum


function ∇mean(dy, ::typeof(Statistics.mean), x::AbstractArray, dims=1:ndims(x))
    dx = similar(x)
    dx .= dy ./ prod(size(x, d) for d in dims)
    return NoTangent(), dx
end
@drule Statistics.mean(x::AbstractArray) ∇mean


## special broadcast

function ∇broadcasted_special(dy, ::typeof(broadcasted), ::typeof(+), x, y)
    return NoTangent(), NoTangent(), unbroadcast(x, dy), unbroadcast(y, dy)
end
@drule broadcasted(::typeof(+), x::AbstractArray, y::AbstractArray) ∇broadcasted_special

function ∇broadcasted_special(dy, ::typeof(broadcasted), ::typeof(*), x, y)
    return NoTangent(), NoTangent(), unbroadcast_prod_x(x, y, dy), unbroadcast_prod_y(x, y, dy)
end
@drule broadcasted(::typeof(*), x::AbstractArray, y::AbstractArray) ∇broadcasted_special

function ∇broadcasted(dy, ::typeof(broadcasted), ::typeof(Base.literal_pow),
    ::typeof(^), x::Any, ::Val{p}) where p
    return NoTangent(), NoTangent(), NoTangent(), (@. p * x ^ (p - 1) * dy), ZeroTangent()
end
@drule broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Any, ::Val) ∇broadcasted

function ∇broadcasted(dy, ::typeof(broadcasted),
    ::typeof(^), x::Any, p::Real)
    return NoTangent(), NoTangent(), (@. p * x ^ (p - 1) * dy), ZeroTangent()
end
@drule broadcasted(::typeof(^), x::Any, ::Real) ∇broadcasted



## __new__, getfield, getproperty

function ∇getproperty(dy, ::typeof(getproperty), s, f::Symbol)
    T = typeof(s)
    nt = NamedTuple{(f,)}((dy,))
    return NoTangent(), Tangent{T}(; nt...), ZeroTangent()
end
@drule getproperty(s::Any, f::Symbol) ∇getproperty


function ∇getfield(dy, ::typeof(getfield), s::Tuple, f::Int)
    T = typeof(s)
    return NoTangent(), Tangent{T}([i == f ? dy : ZeroTangent() for i=1:length(s)]...), ZeroTangent()
end
@drule getfield(s::Tuple, f::Union{Symbol, Int}) ∇getfield


function ∇__new__(dy, ::typeof(__new__), T, args...)
    return NoTangent(), NoTangent(), [getproperty(dy, fld) for fld in fieldnames(T)]...
end
@drule __new__(T::Any, args::Vararg) ∇__new__


## iterate

function ∇iterate(dy, ::typeof(iterate), x::AbstractArray)
    return NoTangent(), ungetindex(x, dy, 1)
end
function ∇iterate(dy, ::typeof(iterate), x::AbstractArray, i::Integer)
    return NoTangent(), ungetindex(x, dy, i), ZeroTangent()
end
@drule iterate(x::AbstractArray) ∇iterate
@drule iterate(x::AbstractArray, i::Integer) ∇iterate

function ∇iterate(dy, ::typeof(iterate), t::Tuple)
    return NoTangent(), ungetfield(dy[1], t, 1)
end
function ∇iterate(dy, ::typeof(iterate), t::Tuple, i::Int)
    return NoTangent(), ungetfield(dy[1], t, i), ZeroTangent()
end
@drule iterate(x::Tuple) ∇iterate
@drule iterate(x::Tuple, i::Integer) ∇iterate

# here we explicitely stop propagation in iteration
# over ranges (e.g for i=1:3 ... end)
function ∇iterate(dy, ::typeof(iterate), x::UnitRange)
    return NoTangent(), ZeroTangent()
end
function ∇iterate(dy, ::typeof(iterate), x::UnitRange, i::Int)
    return NoTangent(), ZeroTangent(), ZeroTangent()
end
@drule iterate(x::UnitRange) ∇iterate
@drule iterate(x::UnitRange, i::Integer) ∇iterate


## tuple unpacking

function ∇indexed_iterate(dy, ::typeof(Base.indexed_iterate), t::Tuple, i::Int)
    return NoTangent(), ungetfield(dy[1], t, i), ZeroTangent()
end
@drule Base.indexed_iterate(t::Tuple, i::Int) ∇indexed_iterate

function ∇indexed_iterate(dy, ::typeof(Base.indexed_iterate), t::Tuple, i::Int, state::Int)
    return NoTangent(), ungetfield(dy[1], t, i), ZeroTangent(), ZeroTangent()
end
@drule Base.indexed_iterate(t::Tuple, i::Int, state::Int) ∇indexed_iterate


## tuple construction

∇tuple(dy, ::typeof(tuple), args...) = (NoTangent(), [dy[i] for i=1:length(args)]...)
@drule tuple(args::Vararg) ∇tuple

## some no diff functions

@drule Core.kwfunc(f::Any) (dy, _, f) -> NoTangent()

## cat & co.

function ∇cat_kw(dy, ::typeof(Core.kwfunc(cat)), kw::Any, ::typeof(cat), arrs...)
    return (
        NoTangent(),
        NoTangent(),
        NoTangent(),
        [uncat(dy, i, arrs...; dims=kw.dims) for i=1:length(arrs)]...
    )
end
@drule Core.kwfunc(cat)(kw::Any, _::typeof(cat), arrs::Vararg) ∇cat_kw

function ∇vcat(dy, ::typeof(vcat), arrs...)
    return NoTangent(), [uncat(dy, i, arrs...; dims=1) for i=1:length(arrs)]...
end
@drule vcat(arrs::Vararg) ∇vcat

function ∇hcat(dy, ::typeof(hcat), arrs...)
    return NoTangent(), [uncat(dy, i, arrs...; dims=2) for i=1:length(arrs)]...
end
@drule hcat(arrs::Vararg) ∇hcat
