function remove_first_parameter(sig)
    if sig isa UnionAll
        new_body = remove_first_parameter(sig.body)
        return UnionAll(sig.var, new_body)
    elseif sig isa DataType
        params = sig.parameters
        return Tuple{params[2:end]...}
    else
        error("Unsupported type: $sig")
    end
end


# sig1 = (@which rrule(sum, rand(3))).sig
# sig2 = (@which rrule(mean, rand(3))).sig


# function rrule_types(meth::Method)
#     sig = deepcopy(meth.sig)
#     subsig = sig
#     while subsig isa UnionAll
#         subsig = subsig.body
#     end
#     # hey, do you know how to corrupt the method table? know I know
#     # deepcopy doesn't work for types, so what am I supposed to do?
#     subsig.parameters = Core.svec(subsig.parameters[2:end])
#     return sig
# end


function chainrules_supported_signatures()
    rrule_methods = methods(rrule).ms
    return [remove_first_parameter(rr.sig) for rr in rrule_methods]
end


const CHAIN_RULE_PRIMITIVES = Ref(FunctionResolver{Bool}())


function update_chainrules_primitives!()
    P = FunctionResolver{Bool}([sig => true for sig in chainrules_supported_signatures()])
    delete!(P.signatures, Symbol(Any))
    CHAIN_RULE_PRIMITIVES[] = P
end


is_chainrules_primitive(sig) = sig in CHAIN_RULE_PRIMITIVES[]


################################################################


# ∇mul(dy, ::typeof(*), x::Number, y::Number) = (NO_FIELDS, dy * y, dy * x)


# x = ...
# y = ...
# dy = 1.0
# r = ∇mul(dy, *, x, y)
# dx = getfield(r, 2)
# dy = getfield(r, 3)



# value, state, deriv function

# function yo_rule(::typeof(broadcasted), ::F, args...) where F
#     f = __new__(F)
#     df = get_yo_rule(F, map(eltype, args)...)
#     return @. df(dy, args...)
# end


# function rrule(::typeof(broadcasted), ::F, args...) where F
#     f = __new__(F)
#     # y = f.(args...)
#     y1, pb = rrule(f, [a[1] for a in args]...)
#     function pullback(Δ)
#         NO_FIELDS, NO_FIELDS, @.(Δ * $df1), @.(Δ * $df2)
#     end
#     return y, $pullback
# end