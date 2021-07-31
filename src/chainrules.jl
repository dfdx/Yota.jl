import ChainRulesCore: rrule, rrule_via_ad, RuleConfig, NoForwardsMode, HasReverseMode

###############################################################################
#                              Primitives                                     #
###############################################################################

function chainrules_supported_signatures()
    rrule_methods = methods(rrule).ms
    sigs = [remove_first_parameter(rr.sig) for rr in rrule_methods]
    # add keyword version of these functions as well
    kw_sigs = [kwsig for kwsig in map(kwfunc_signature, sigs) if kwsig !== Tuple{}]
    return [sigs; kw_sigs]
end


const CHAIN_RULE_PRIMITIVES = Ref(FunctionResolver{Bool}())


function update_chainrules_primitives!()
    P = FunctionResolver{Bool}([sig => true for sig in chainrules_supported_signatures()])
    delete!(P.signatures, Symbol(Any))
    CHAIN_RULE_PRIMITIVES[] = P
end


is_chainrules_primitive(sig) = sig in CHAIN_RULE_PRIMITIVES[]


###############################################################################
#                              RuleConfig                                     #
###############################################################################

struct YotaRuleConfig <: RuleConfig{Union{NoForwardsMode, HasReverseMode}} end


function value_and_pullback(f, args...)
    # TODO: make back!() accept seed=:auto, translated into
    # ones(size(tape[tape.result].val))
    tape = gradtape(f, args...)
    val = tape[tape.result].val
    grad_fn = grad_compile(tape)
    function generic_yota_pullback(dy)
        return grad_fn(f, args...; seed=dy)[2]
    end
    return val, generic_yota_pullback
end

function ChainRulesCore.rrule_via_ad(::YotaRuleConfig, f, args...)
    return value_and_pullback(f, args...)
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
