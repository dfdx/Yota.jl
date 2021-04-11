function rrule_types(rr::Method)
    sig = rr.sig
    while sig isa UnionAll
        sig = sig.body
    end
    return tuple(collect(sig.parameters[2:end])...)
end


function rrule_signatures()
    rrules = methods(rrule).ms
    return [rrule_types(rr) for rr in rrules]
end


const CHAIN_RULE_PRIMITIVES = Ref(FunctionResolver{Bool}())


function update_chain_rules!()
    P = FunctionResolver{Bool}([sig => true for sig in rrule_signatures()])
    delete!(P.signatures, Any)
    CHAIN_RULE_PRIMITIVES[] = P
end


is_chainrules_primitive(sig) = sig in CHAIN_RULE_PRIMITIVES[]
