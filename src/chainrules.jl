import ChainRules.rrule


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