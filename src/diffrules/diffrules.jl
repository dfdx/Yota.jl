########################################################################
#                               API                                    #
########################################################################

const DIFF_RULES = Vector{Tuple}()
const NO_DIFF_RULES = Vector{Tuple}()
const CONSTRUCTORS = Vector{Tuple}()
const DIFF_PHS = Set([:x, :u, :v, :w, :t, :i, :j, :k,])


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
        resolve_functions_and_types!(mod, @view ex.args[2:end])
        # for x in ex.args[2:end]
        #     resolve_functions_and_types!(mod, x)
        # end
    elseif Meta.isexpr(ex, :(::))
        ex.args[2] = Core.eval(mod, ex.args[2])
    elseif ex isa Vector || ex isa SubArray
        for (i, x) in enumerate(ex)
            if Meta.isexpr(x, :$)
                ex[i] = Core.eval(mod, x.args[1])
            else
                resolve_functions_and_types!(mod, x)
            end
        end
    end
    return ex
end


add_diff_rule(rule) = push!(DIFF_RULES, rule)
add_primitive(primitive) = push!(PRIMITIVES, primitive)
add_constructor(ctor) = push!(CONSTRUCTORS, ctor)


"""
Define a differentiation rule for a function.

Arguments:

 * pat - pattern of a function call, may include argument types
 * var - variable (argument) for which the rule is defined
 * rpat - replacement pattern, i.e. expression that calculates the derivative

Examples:

    @diffrule +(u::Real, v::Real) u dy
    @diffrule sum(u::AbstractArray) u sum_grad(u, dy)
    @diffrule Statistics.mean(u::AbstractArray) u ∇mean(u, dy)
    @diffrule logpdf(_d::MvNormal, x) _d.μ ∇logpdf(dy, _d, x)

Note that diff rules are added dynamically, so it's advised to put @diffrule
to __init__() method of a module to survive module precompilation.

Allowed and special variable names:

 * x, u, v, w and some others (see Yota.DIFF_PHS for the full list)
 * any var beginning with _, e.g. _a, _d, etc.
 * y - special, replaced with the result of the function call (e.g. `y = func(args)`)
 * dy - special, replaced with the derivative w.r.t. to y

See also: @diffrule_kw, @nodiff, @ctor
"""
macro diffrule(pat, var, rpat)
    esc(quote
        mod = @__MODULE__
        local pat, rpat = map($Espresso.sanitize, ($(QuoteNode(pat)), $(QuoteNode(rpat))))
        pat, rpat = map($resolve_old_broadcast, (pat, rpat))
        $resolve_functions_and_types!(mod, pat)
        $resolve_functions_and_types!(mod, rpat)
        $add_diff_rule((pat, $(QuoteNode(var)), rpat))
        $add_primitive(pat.args[1])
        nothing
        end)
end


"""
Define a differentiation rule for a functioin with keyword arguments.
Keyword arguments need not to be specified; instead all kw args are passed to
the gradient functions as is.

Examples:

    conv2d(x, w; stride=1, padding=0, dilation=1) = ...
    ∇conv2d_w(dy, x, w; stride=1, padding=0, dilation=1) = ...
    @diffrule_kw conv2d(x, w) w ∇conv2d_w(dy, x, w)
    @diffrule_kw conv2d(x, w) x ∇conv2d_x(dy, x, w)  # kwargs passed implicitely

See also: @diffrule, @nodiff, @ctor

"""
macro diffrule_kw(pat, var, rpat)
    kw_pat = rewrite(pat, :(_fn(_args...)), :(Core.kwfunc(_fn)(_kw, _, _args...)))
    kw_rpat = rewrite(rpat, :(_fn(_args...)), :(Core.kwfunc(_fn)(_kw, _, _args...)))
    kw_rpat = subs(kw_rpat, Dict(:_ => Expr(:$, rpat.args[1])))
    return esc(
        quote
        @diffrule $pat $var $rpat
        @diffrule $kw_pat $var $kw_rpat
        @nodiff $kw_pat _kw
        @nodiff $kw_pat _
        end)
end


"""
Define a type constructor that should not be traced, but instead recorded
to the tape as is. Here's an example:

    @ctor MvNormal(μ, Σ)    

This should be read as:

1) record MvNormal to the tape as is
2) if @diffrule w.r.t. its field is added, propagate it directly to the variable
   that was used to construct that field

For example, after the following diff rule:

    @diffrule logpdf(_d::MvNormal, x) _d.μ ∇logpdf(dy, _d, x)

Yota will _completely bypass_ internals of the constructor and jump directly to the
1st variable passed to MvNormal().

Note that if you don't want to bypass the constructor (which you normally shouldn't do),
you can rely on Yota handling it automatically. 

"""
macro ctor(ex)
    esc(quote
        mod = @__MODULE__
        local ex = $Espresso.sanitize($(QuoteNode(ex)))
        ex = $resolve_old_broadcast(ex)
        $resolve_functions_and_types!(mod, ex)
        $add_constructor((ex.args[1], ex.args[2:end]...))
        $add_primitive(ex.args[1])
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
    # if var name is actually a field expression, e.g. :(str.fld)
    fldname = nothing
    if Meta.isexpr(vname, :(.))
        fldname = vname.args[2].value
        vname = vname.args[1]
    end
    vidx = findfirst(isequal(vname), get_arg_names(tpat))
    if idx != vidx
        return nothing, nothing
    end
    pat_types = get_arg_types(tpat)
    if (length(dep_types) != length(pat_types) ||
        !all([t <: p for (t, p) in zip(dep_types, pat_types)]))
        return nothing, nothing
    end
    pat = without_types(tpat)
    ex_ = without_keywords(ex)
    if !matchingex(pat, ex_; phs=DIFF_PHS)
        return nothing, nothing
    else
        return pat => rpat, fldname
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
For each matched rule, rewrite call expression into its derivative w.r.t.
its argument with index `idx`. If argument under `idx` is a struct,
second returned values is a symbolic name of the field that this derivative
should be attributed to. Otherwise it's nothing.

Example:

```
deriv_exprs(:($sin(x)), [Float64], 1)
# ==> :((cos(x) * ds)), nothing
```
"""
function deriv_exprs(ex, dep_types, idx::Int)
    result = Tuple[]   # list of tuples (rewritten_expr, field_name | nothing)
    for rule in DIFF_RULES
        m, fldname = match_rule(rule, ex, dep_types, idx)
        if m != nothing
            pat, rpat = m
            rex = rewrite_with_keywords(ex, pat, rpat)
            push!(result, (rex, fldname))
        end
    end
    if isempty(result)
        error("Can't find differentiation rule for $ex at $idx " *
              "with types $dep_types)")
    end
    return result
end



"""
Internal function for finding rule by function name
"""
function find_rules_for(fun)
    return [r for r in DIFF_RULES if r[1].args[1] == fun]
end


## nodiff

add_no_diff_rule(rule) = push!(NO_DIFF_RULES, rule)


"""
Don't propagate derivative of the given function w.r.t. specified variable
"""
macro nodiff(pat, var)
    esc(quote
        mod = @__MODULE__
        local pat = $Espresso.sanitize($(QuoteNode(pat)))
        pat = $resolve_old_broadcast(pat)
        $resolve_functions_and_types!(mod, pat)
        $add_no_diff_rule((pat, $(QuoteNode(var))))
        $add_primitive(pat.args[1])
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
    if op.fn == broadcast
        ex = to_unbroadcast_expr(tape, op)
        dep_types = [eltype(tape[arg].typ) for arg in op.args[2:end]]
        idx_ = idx - 1
    else
        ex =  to_expr(tape, op)
        dep_types = [tape[arg].typ for arg in op.args]
        idx_ = idx
    end
    for rule in NO_DIFF_RULES
        if match_nodiff_rule(rule, ex, dep_types, idx_)
            return true
        end
    end
    return false
end


include("basic.jl")
