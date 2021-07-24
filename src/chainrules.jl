import ChainRulesCore: rrule, rrule_via_ad, RuleConfig, NoForwardsMode, HasReverseMode
import Ghost: make_name, Input, to_expr

###############################################################################
#                              Primitives                                     #
###############################################################################

function function_signatures(fn)
    rrule_methods = methods(fn).ms
    sigs = [remove_first_parameter(rr.sig) for rr in rrule_methods]
    # add keyword version of these functions as well
    kw_sigs = [kwsig for kwsig in map(kwfunc_signature, sigs) if kwsig !== Tuple{}]
    return [sigs; kw_sigs]
end


const CHAINRULES_PRIMITIVES = Ref(FunctionResolver{Bool}())
const NUM_CHAINRULES_METHODS = Ref{Int}(0)


function update_chainrules_primitives!(;force=false)
    num_methods = length(methods(rrule)) + length(methods(no_rrule))
    if force || num_methods != NUM_CHAINRULES_METHODS[]
        sigs_flags = [
            [sig => true for sig in function_signatures(rrule)];
            [sig => false for sig in function_signatures(no_rrule)]  # override rrule(sig...)
            ]
        P = FunctionResolver{Bool}(sigs_flags)
        CHAINRULES_PRIMITIVES[] = P
        NUM_CHAINRULES_METHODS[] = num_methods
    end
end


is_chainrules_primitive(sig) = sig in CHAIN_RULE_PRIMITIVES[]


###############################################################################
#                              RuleConfig                                     #
###############################################################################

struct YotaRuleConfig <: RuleConfig{Union{NoForwardsMode, HasReverseMode}} end


function to_rrule_expr(tape::Tape)
    fn_name = gensym("rrule_$(tape[V(1)].val)")
    header = Expr(:call, fn_name)
    for v in inputs(tape)
        op = tape[v]
        push!(header.args, Expr(:(::), make_name(op), op.typ))
    end
    body = Expr(:block)
    # generate transformed forward pass
    seed_id = tape.meta[:seed].id
    for op in tape.ops[1:seed_id - 1]
        op isa Input && continue
        ex = to_expr(op)
        if ex isa Vector
            push!(body.args, ex...)
        else
            push!(body.args, ex)
        end
    end
    # generate pullback
    pb_name = gensym("pullback_$(tape[V(1)].val)")
    pb_ex = :(function $pb_name(dy) end)
    pb_body = pb_ex.args[2]
    empty!(pb_body.args)  # clean from useless linenumber nodes
    push!(pb_body.args, Expr(:(=), make_name(tape.meta[:seed].id), :dy))
    for op in tape.ops[seed_id + 1:length(tape) - 2]
        op isa Input && continue
        ex = to_expr(op)
        if ex isa Vector
            push!(pb_body.args, ex...)
        else
            push!(pb_body.args, ex)
        end
    end
    push!(body.args, pb_ex)
    # generate return
    result_name = make_name(tape[tape.result].args[1].id)
    push!(body.args, Expr(:tuple, result_name, pb_name))
    fn_ex = Expr(:function, header, body)
    return fn_ex
end


"""
    make_rrule(tape::Tape)
    make_rrule(f, args...)

Generate a function equivalent to (but not extending) ChainRulesCore.rrule(),
i.e. returning the primal value and the pullback.


### Examples:

```
foo(x) = 2x + 1
rr = make_rrule(foo, 2.0)
val, pb = rr(foo, 3.0)
pb(1.0)
```
"""
make_rrule(tape::Tape) = Base.eval(@__MODULE__, to_rrule_expr(tape))
make_rrule(f, args...) = make_rrule(gradtape(f, args...))


const GENERATED_RRULE_CACHE = Dict()

function ChainRulesCore.rrule_via_ad(::YotaRuleConfig, f, args...)
    sig = call_signature(f, args...)
    if haskey(GENERATED_RRULE_CACHE, sig)
        rr = GENERATED_RRULE_CACHE[sig]
        return Base.invokelatest(rr, f, args...)
    else
        rr = make_rrule(f, args...)
        GENERATED_RRULE_CACHE[sig] = rr
        return Base.invokelatest(rr, f, args...)
    end
end

###############################################################################
#                                 Rules                                       #
###############################################################################

function ChainRulesCore.rrule(::typeof(broadcasted), ::typeof(identity), x)
    identity_pullback(dy) = (NoTangent(), NoTangent(), dy)
    return x, identity_pullback
end

# test_rrule(broadcasted, identity, [1.0, 2.0]) -- fails at the moment


function ChainRulesCore.rrule(
    ::typeof(Core._apply_iterate), ::typeof(iterate), f::F, args...) where F
    flat = []
    for a in args
        push!(flat, a...)
    end
    return rrule(f, flat...)
end


function ChainRules.rrule(::typeof(tuple), args...)
    y = tuple(args...)
    return y, dy -> (NoTangent(), dy...)
end

# test_rrule(tuple, 1, 2, 3)
