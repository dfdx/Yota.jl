
function rrule_types(rr::Method)
    sig = rr.sig
    while sig isa UnionAll
        sig = sig.body
    end
    return tuple(collect(sig.parameters[2:end])...)
end


function rrule_primitives()
    rrules = methods(rrule).ms
    return [rrule_types(rr) for rr in rrules]
end