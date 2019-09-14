########################################################################
#                               API                                    #
########################################################################

const DIFF_RULES = Vector{Tuple}()
const NO_DIFF_RULES = Vector{Tuple}()
const DIFF_PHS = Set([:x, :u, :v, :w, :i, :j, :k,])


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

add_no_diff_rule(rule) = push!(NO_DIFF_RULES, rule)

macro nodiff(pat, var)
    esc(quote
        mod = @__MODULE__
        local pat = $Espresso.sanitize($(QuoteNode(pat)))
        pat, $resolve_old_broadcast(pat)
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
    ex = to_expr(tape, op)
    dep_types = [tape[arg].typ for arg in op.args]
    for rule in NO_DIFF_RULES
        if match_nodiff_rule(rule, ex, dep_types, idx)
            return true
        end
    end
    return false
end


include("basic.jl")
