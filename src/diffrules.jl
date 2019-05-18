########################################################################
#                               API                                    #
########################################################################

const DIFF_RULES = Vector{Tuple}()
const NO_DIFF_RULES = Vector{Tuple}()
const DIFF_PHS = Set([:w, :x, :y, :z, :i, :j, :k,])


function resolve_old_broadcast(ex)
    # rewrite dot op symbols like .*, .+, etc. into broadcasting
    for (pat, rpat) in [
        :(.+(_xs...)) => :($broadcast($+, _xs...)),
        :(.-(_xs...)) => :($broadcast($-, _xs...)),
        :(.*(_xs...)) => :($broadcast($*, _xs...)),
        :(./(_xs...)) => :($broadcast($/, _xs...)),
    ]
        ex = Espresso.rewrite_all(ex, pat, rpat)
    end
    return ex
end


function resolve_functions_and_types!(mod::Module, ex)
    if Meta.isexpr(ex, :call)
        # replace function symbol with actual function reference and
        # recursively call on function args
        if string(ex.args[1])[1] != '.'  # .*, .+, etc.
            ex.args[1] = Core.eval(mod, ex.args[1])
        end
        for x in ex.args[2:end]
            resolve_functions_and_types!(mod, x)
        end
    elseif Meta.isexpr(ex, :(::))
        ex.args[2] = Core.eval(mod, ex.args[2])
    elseif ex isa Vector
        for x in ex
            resolve_functions_and_types!(mod, x)
        end
    end
    return ex
end


# not used?
function add_diff_rule(mod, pat, var, rpat)
    pat, rpat = map(Espresso.sanitize, (pat, rpat))
    resolve_functions_and_types!(mod, pat)
    resolve_functions_and_types!(mod, rpat)
    push!(DIFF_RULES, (pat, var, rpat))
    push!(PRIMITIVES, pat.args[1])
end

macro diffrule(pat, var, rpat)
    esc(quote
        mod = @__MODULE__
        local pat, rpat = map($Espresso.sanitize, ($(QuoteNode(pat)), $(QuoteNode(rpat))))
        pat, rpat = map($resolve_old_broadcast, (pat, rpat))
        $resolve_functions_and_types!(mod, pat)
        $resolve_functions_and_types!(mod, rpat)
        push!($DIFF_RULES, (pat, $(QuoteNode(var)), rpat))
        push!($PRIMITIVES, pat.args[1])
        nothing
        end)
end


isparameters(a) = isa(a, Expr) && a.head == :parameters

function without_types(pat)
    rpat = copy(pat)
    for i=2:length(rpat.args)
        a = rpat.args[i]
        if !isparameters(a)  # parameters aren't modified
            rpat.args[i] = isa(a, Expr) ? a.args[1] : a
        end
    end
    return rpat
end


function get_arg_names(pat)
    return [isa(a, Expr) ? a.args[1] : a for a in pat.args[2:end] if !isparameters(a)]
end

function get_arg_types(pat)
    return [isa(a, Expr) ? Core.eval(Base, a.args[2]) : Any
            for a in pat.args[2:end] if !isparameters(a)]
end


function match_rule(rule, ex, dep_types, idx)
    tpat, vname, rpat = rule
    vidx = findfirst(isequal(vname), get_arg_names(tpat))
    if idx != vidx
        return nothing
    end
    pat_types = get_arg_types(tpat)
    if (length(dep_types) != length(pat_types) ||
        !all([t <: p for (t, p) in zip(dep_types, pat_types)]))
        return nothing
    end
    pat = without_types(tpat)
    ex_ = without_keywords(ex)
    if !matchingex(pat, ex_; phs=DIFF_PHS)
        return nothing
    else
        return pat => rpat
    end
end


function rewrite_with_keywords(ex, pat, rpat)
    if rpat isa Expr && rpat.head == :call
        op, args, kw_args = parse_call_expr(ex)
        ex_no_kw = Expr(:call, op, args...)
        rex_no_kw = rewrite(ex_no_kw, pat, rpat; phs=DIFF_PHS)
        rex = with_keywords(rex_no_kw, kw_args)
    else
        rex = rewrite(ex, pat, rpat; phs=DIFF_PHS)
    end
    return rex
end


"""
Rewrite single call expression into its derivative. Example:

```
deriv_expr(:($sin(x)), [Float64], 1)
# ==> :((cos(x) * ds))
```
"""
function deriv_expr(ex, dep_types, idx::Int)
    rex = nothing
    for rule in DIFF_RULES
        m = match_rule(rule, ex, dep_types, idx)
        if m != nothing
            pat, rpat = m
            rex = rewrite_with_keywords(ex, pat, rpat)
            break
        end
    end
    if rex == nothing
        error("Can't find differentiation rule for $ex at $idx " *
              "with types $dep_types)")
    end
    return rex
end



"""
Internal function for finding rule by function name
"""
function find_rules_for(fun)
    return [r for r in DIFF_RULES if r[1].args[1] == fun]
end


## nodiff

macro nodiff(pat, var)
    esc(quote
        mod = @__MODULE__
        local pat = $Espresso.sanitize($(QuoteNode(pat)))
        pat, $resolve_old_broadcast(pat)
        $resolve_functions_and_types!(mod, pat)
        push!($NO_DIFF_RULES, (pat, $(QuoteNode(var))))
        push!($PRIMITIVES, pat.args[1])
        nothing
        end)
end


function match_nodiff_rule(rule, ex, dep_types, idx)
    tpat, vname = rule
    vidx = findfirst(isequal(vname), get_arg_names(tpat))
    if idx != vidx
        return false
    end
    pat_types = get_arg_types(tpat)
    if (length(dep_types) != length(pat_types) ||
        !all([t <: p for (t, p) in zip(dep_types, pat_types)]))
        return false
    end
    pat = without_types(tpat)
    ex_ = without_keywords(ex)
    return matchingex(pat, ex_; phs=DIFF_PHS)
end


function dont_diff(tape::Tape, op::AbstractOp, idx::Int)
    ex = to_expr(tape, op)
    dep_types = [tape[arg].typ for arg in op.args]
    for rule in NO_DIFF_RULES
        if match_nodiff_rule(rule, ex, dep_types, idx)
            return true
        end
    end
    return false
end


#######################################################################
#                              RULES                                  #
#######################################################################

# derivation neutral functions
# @diffrule colon(x,y)   x     0.0
# @diffrule colon(x,y)   y     0.0

@diffrule length(x)    x     0.0

@diffrule size(x)      x     0.0
@diffrule size(x,y)    x     0.0
@diffrule size(x,y)    y     0.0

@diffrule fill(x,y)    x     0.0
@diffrule fill(x,y)    y     0.0

@diffrule similar(x,y) x     0.0
@diffrule similar(x,y) y     0.0

@diffrule zeros(x)     x     0.0

@diffrule ones(x)      x     0.0

# @diffrule cell(x)      x     0.0

@diffrule sign(x)      x     0.0

@diffrule reverse(x)   x     0.0

# tuple
@diffrule tuple(x)        x     ds[1]
@diffrule tuple(x,y)      x     ds[1]
@diffrule tuple(x,y)      y     ds[2]
@diffrule tuple(x,y,z)    x     ds[1]
@diffrule tuple(x,y,z)    y     ds[2]
@diffrule tuple(x,y,z)    z     ds[3]
@diffrule tuple(x,y,z,t)  x     ds[1]
@diffrule tuple(x,y,z,t)  y     ds[2]
@diffrule tuple(x,y,z,t)  z     ds[3]
@diffrule tuple(x,y,z,t)  t     ds[4]

# __tuple__ (current tracer implemetation uses it instead of normal tuple)
@diffrule __tuple__(x)        x     ds[1]
@diffrule __tuple__(x,y)      x     ds[1]
@diffrule __tuple__(x,y)      y     ds[2]
@diffrule __tuple__(x,y,z)    x     ds[1]
@diffrule __tuple__(x,y,z)    y     ds[2]
@diffrule __tuple__(x,y,z)    z     ds[3]
@diffrule __tuple__(x,y,z,t)  x     ds[1]
@diffrule __tuple__(x,y,z,t)  y     ds[2]
@diffrule __tuple__(x,y,z,t)  z     ds[3]
@diffrule __tuple__(x,y,z,t)  t     ds[4]

#  vcat
@diffrule vcat(x)        x     ds[1]
@diffrule vcat(x,y)      x     ds[1]
@diffrule vcat(x,y)      y     ds[2]
@diffrule vcat(x,y,z)    x     ds[1]
@diffrule vcat(x,y,z)    y     ds[2]
@diffrule vcat(x,y,z)    z     ds[3]
@diffrule vcat(x,y,z,t)  x     ds[1]
@diffrule vcat(x,y,z,t)  y     ds[2]
@diffrule vcat(x,y,z,t)  z     ds[3]
@diffrule vcat(x,y,z,t)  t     ds[4]

# reshape
@diffrule reshape(x::AbstractArray, _a)             x    reshape(ds, size(x))
@diffrule reshape(x::AbstractArray, _a)             _a   zero(eltype(x))
@diffrule reshape(x::AbstractArray, _a, _b)         x    reshape(ds, size(x))
@diffrule reshape(x::AbstractArray, _a, _b)        _a    zero(eltype(x))
@diffrule reshape(x::AbstractArray, _a, _b)        _b    0
@diffrule reshape(x::AbstractArray, _d::Tuple)      x    reshape(ds, size(x))
@diffrule reshape(x::AbstractArray, _d::Tuple)     _d    0

@diffrule vec(x::AbstractArray)    x    reshape(ds, size(x))


@diffrule getindex(x::AbstractArray, i)         x    ungetindex(x, ds, i)
@diffrule getindex(x::AbstractArray, i, j)      x    ungetindex(x, ds, i, j)
@diffrule getindex(x::AbstractArray, i, j, k)   x    ungetindex(x, ds, i, j, k)
@diffrule getindex(x::AbstractArray, i)         i    0
@diffrule getindex(x::AbstractArray, i, j)      i    0
@diffrule getindex(x::AbstractArray, i, j)      j    0
@diffrule getindex(x::AbstractArray, i, j, k)   i    0
@diffrule getindex(x::AbstractArray, i, j, k)   j    0
@diffrule getindex(x::AbstractArray, i, j, k)   k    0


# square root
@diffrule sqrt(x::Real)              x     one(x) / 2 * x ^ (-one(x) / 2) * ds

# addition
@diffrule +(x::Real         , y::Real )            x     ds
@diffrule +(x::AbstractArray, y::AbstractArray)    x     ds
@diffrule +(x::Real         , y::Real )            y     ds
@diffrule +(x::AbstractArray, y::AbstractArray)    y     ds

@diffrule broadcast(_fn::typeof(+), x, y) x unbroadcast(x, ds)
@diffrule broadcast(_fn::typeof(+), x, y) y unbroadcast(y, ds)

@diffrule broadcast(_fn::typeof(*), x, y) x unbroadcast_prod_x(x, y, ds)
@diffrule broadcast(_fn::typeof(*), x, y) y unbroadcast_prod_y(x, y, ds)


# unary substraction
@diffrule -(x::Real )                               x     -ds
@diffrule -(x::AbstractArray)                       x     -ds

# binary substraction
@diffrule -(x::Real, y::Real)                       x     ds
# @diffrule -(x::Real, y::AbstractArray)              x     sum(ds)
# @diffrule -(x::AbstractArray, y::Real)              x     ones(size(x)) .* ds
@diffrule -(x::AbstractArray, y::AbstractArray)     x     ds
@diffrule -(x::Real         , y::Real)              y     -ds
# @diffrule -(x::Real, y::AbstractArray)              y     -ones(size(y)) .* ds
# @diffrule -(x::AbstractArray, y::Real)              y     -sum(ds)
@diffrule -(x::AbstractArray, y::AbstractArray)     y     -ds


# sum() and mean()
# @diffrule sum(x::Real)                              x     ds
@diffrule sum(x::AbstractArray)                     x     sum_grad(x, ds)
@diffrule Base._sum(x::AbstractArray, y::Int)             x     sum_grad(x, ds)
@diffrule Base._sum(x::AbstractArray, y::Int)             y     zero(eltype(x))

# special sums
@diffrule sum(_fn::typeof(log), x::AbstractArray)    x    sum_grad(x, ds) ./ x

# @diffrule Statistics.mean(x::Real)                         x     ds
@diffrule Statistics.mean(x::AbstractArray)                x     mean_grad(x, ds)
@diffrule Statistics._mean(x::AbstractArray, y::Int)       x     mean_grad(x, ds)
@diffrule Statistics._mean(x::AbstractArray, y::Int)       y     zero(eltype(x))

# diag
@diffrule diag(x::AbstractMatrix)    x    Diagonal(ds)

# dot()
@diffrule dot(x::Real, y::Real)                     x     y * ds
@diffrule dot(x::Real, y::Real)                     y     x * ds

# @diffrule dot(x::AbstractArray, y::AbstractArray)   x     y.*ds
# @diffrule dot(x::AbstractArray, y::AbstractArray)   y     x.*ds

# log() and exp()
@diffrule log(x::Real )                            x     ds / x
@diffrule exp(x::Real )                            x     exp(x) * ds
@diffrule log1p(x::Real)                           x     ds  / (one(x) + x)
@diffrule expm1(x::Real)                           x     (one(x) + expm1(x))  * ds
# @diffrule expm1(x::AbstractArray)                  x     (one(x) + expm1(x)) .* ds
# note : derivative uses expm1() and not exp() to reuse the
#   already calculated expm1()

# trig functions
@diffrule sin(x::Real )                            x     cos(x) * ds
@diffrule cos(x::Real )                            x     -sin(x) * ds
@diffrule tan(x::Real )                            x     (one(x) + tan(x)  * tan(x))  * ds
@diffrule sinh(x::Real )                           x     cosh(x) * ds
@diffrule cosh(x::Real )                           x     sinh(x) * ds
@diffrule tanh(x::Real )                           x     (one(x) - tanh(x)  * tanh(x))  * ds
@diffrule asin(x::Real )                           x     ds  / sqrt(one(x) - x*x)
@diffrule acos(x::Real )                           x     -ds  / sqrt(one(x) - x*x)
@diffrule atan(x::Real )                           x     ds  / (one(x) + x *x)


# round, floor, ceil, trunc, mod2pi
@diffrule round(x::Real)                           x     zero(x)

@diffrule floor(x::Real)                           x     zero(x)

@diffrule ceil(x::Real)                            x     zero(x)

@diffrule trunc(x::Real)                           x     zero(x)

@diffrule mod2pi(x::Real)                          x     ds


# abs, max(), min()
@diffrule abs(x::Real)                             x     sign(x) * ds
@diffrule abs2(x::Real)                            x     2 * x * ds


# @diffrule max(x::Real         , y::Real)           x     (x > y) * ds
# @diffrule max(x::Real         , y::AbstractArray)  x     sum((x .> y) .* ds)
# @diffrule max(x::AbstractArray, y::Real)           x     (x .> y) .* ds
# @diffrule max(x::AbstractArray, y::AbstractArray)  x     (x .> y) .* ds

# @diffrule max(x::Real         , y::Real)           y     (x < y) * ds
# @diffrule max(x::Real         , y::AbstractArray)  y     (x .< y) .* ds
# @diffrule max(x::AbstractArray, y::Real)           y     sum((x .< y) .* ds)
# @diffrule max(x::AbstractArray, y::AbstractArray)  y     (x .< y) .* ds

# @diffrule min(x::Real         , y::Real)           x     (x < y) * ds
# @diffrule min(x::Real         , y::AbstractArray)  x     sum((x .< y) .* ds)
# @diffrule min(x::AbstractArray, y::Real)           x     (x .< y) .* ds
# @diffrule min(x::AbstractArray, y::AbstractArray)  x     (x .< y) .* ds

# @diffrule min(x::Real         , y::Real)           y     (x > y) * ds
# @diffrule min(x::Real         , y::AbstractArray)  y     (x .> y) .* ds
# @diffrule min(x::AbstractArray, y::Real)           y     sum((x .> y) .* ds)
# @diffrule min(x::AbstractArray, y::AbstractArray)  y     (x .> y) .* ds

# maximum, minimum
# @diffrule maximum(x::Real         )     x     ds
# @diffrule maximum(x::AbstractArray)     x     (x .== maximum(x)) .* ds

# @diffrule minimum(x::Real         )     x     ds
# @diffrule minimum(x::AbstractArray)     x     (x .== minimum(x)) .* ds


# multiplication
@diffrule *(x::Real         , y::Real )            x     y * ds
# @diffrule *(x::Real         , y::AbstractArray)    x     sum(y .* ds)
# @diffrule *(x::AbstractArray, y::Real )            x     y .* ds
@diffrule *(x::AbstractArray, y::AbstractArray)    x     ds * transpose(y)

@diffrule *(x::Real         , y::Real )            y     x * ds
# @diffrule *(x::Real         , y::AbstractArray)    y     x .* ds
@diffrule *(x::AbstractArray, y::Real )            y     sum(x .* ds)
@diffrule *(x::AbstractArray, y::AbstractArray)    y     transpose(x) * ds


# power  (both args reals)
@diffrule ^(x::Real, y::Real)                      x     y * x ^ (y-(one(x))) * ds
@diffrule ^(x::Real, y::Real)                      y     log(x) * x ^ y * ds
# @diffrule Base.literal_pow(_fn::typeof(^), x::Real, ::Val{y}) x (y * x ^ (y-1) * ds)


# # division
@diffrule /(x::Real          , y::Real )           x     ds / y
@diffrule /(x::Real          , y::Real )           y     -x * ds / (y * y)

# transpose
@diffrule transpose(x::AbstractVector)             x     untranspose_vec(ds)
@diffrule transpose(x::AbstractArray)              x     transpose(ds)
@diffrule adjoint(x::AbstractVector)             x     untranspose_vec(ds)
@diffrule adjoint(x::AbstractArray)              x     adjoint(ds)

# # erf, erfc, gamma, beta, lbeta, lgamma
# @diffrule erf(x::Real)                       x     2.0/sqrt(π) * exp(-x  * x)  * ds
# @diffrule erf(x::AbstractArray)              x     2.0/sqrt(π) .* exp(-x .* x) .* ds

# @diffrule erfc(x::Real)                      x     -2.0/sqrt(π) * exp(-x  * x)  * ds
# @diffrule erfc(x::AbstractArray)             x     -2.0/sqrt(π) .* exp(-x .* x) .* ds

# @diffrule gamma(x::Real)                     x     polygamma(0,x)  * gamma(x)  * ds
# @diffrule gamma(x::AbstractArray)            x     polygamma(0,x) .* gamma(x) .* ds

# @diffrule lgamma(x::Real)                    x     polygamma(0,x)  * ds
# @diffrule lgamma(x::AbstractArray)           x     polygamma(0,x) .* ds

# @diffrule beta(x::Real         , y::Real)            x   beta(x,y)  * (digamma(x)-digamma(x+y))  * ds
# @diffrule beta(x::AbstractArray, y::AbstractArray)   x   beta(x,y) .* (digamma(x)-digamma(x+y)) .* ds
# @diffrule beta(x::Real         , y::Real)            y   beta(x,y)  * (digamma(y)-digamma(x+y))  * ds
# @diffrule beta(x::AbstractArray, y::AbstractArray)   y   beta(x,y) .* (digamma(y)-digamma(x+y)) .* ds

# @diffrule lbeta(x::Real         , y::Real)            x   (polygamma(0,x)-polygamma(0,x+y))  * ds
# @diffrule lbeta(x::AbstractArray, y::AbstractArray)   x   (polygamma(0,x)-polygamma(0,x+y)) .* ds
# @diffrule lbeta(x::Real         , y::Real)            y   (polygamma(0,y)-polygamma(0,x+y))  * ds
# @diffrule lbeta(x::AbstractArray, y::AbstractArray)   y   (polygamma(0,y)-polygamma(0,x+y)) .* ds


# should be something like @nodiff instead
@diffrule rand(x) x zero(x)
