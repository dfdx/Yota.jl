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


################################################################

# from Zygote:
# https://github.com/FluxML/Zygote.jl/blob/d5be4d5ca80e79278d714eaac15ca71904a262e3/src/lib/array.jl#L177-L185
struct StaticGetter{i} end
(::StaticGetter{i})(v) where {i} = v[i]

@generated function _unzip(tuples, ::Val{N}) where {N}
  Expr(:tuple, (:(map($(StaticGetter{i}()), tuples)) for i ∈ 1:N)...)
end

function unzip(tuples)
  N = length(first(tuples))
  _unzip(tuples, Val(N))
end


function rrule(::typeof(Broadcast.broadcasted), f::F, args...) where F
    ys, pbs = unzip(rrule.(f, args...))
    function pullback(Δ)
        dxs = map((pb, Δ) -> pb(Δ), pbs, Δ)
        return NO_FIELDS, unzip(dxs)...
    end
    return ys, pullback
end
