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


# function rrule(::typeof(Broadcast.broadcasted), f::F, args...) where F
#     # y = f.(args...)
#     y1, pb = rrule(f, [a[1] for a in args]...)
#     function pullback(Δ)
#         NO_FIELDS, NO_FIELDS # , @.(Δ * $df1), @.(Δ * $df2)
#     end
#     return y, pullback
# end


# function rrule(::typeof(Broadcast.broadcasted), f::F, args...) where F
#     df = get_deriv_function(Tuple{F, map(eltype, args)...})
#     # TODO: write a more descriptive error message

#     darg_tuples = df.(dy, f, args...)
#     darg_single_tuple = tuple(
#         [vcat([t[i] for t in darg_tuples]) for i in 1:length(args) + 1]...
#     )
#     return NO_FIELDS, darg_single_tuple
# end