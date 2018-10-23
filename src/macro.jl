
## @primitive

function primitive_sig(sig)
    fn, args = sig.args[1], sig.args[2:end]
    targs = Array{Any}(undef, length(args))
    vnames = Array{Symbol}(undef, length(args))
    for i=1:length(args)
        if args[i] isa Expr
            name, typ_ex = args[i].args
            typ = Core.eval(Base, typ_ex)
            ttyp = typ <: AbstractArray ? TArray : TReal
            targs[i] = :($name :: $ttyp)
            vnames[i] = name
        elseif args[i] isa Symbol
            name = args[i]
            targs[i] = :($name :: $TAny)
            vnames[i] = name
        else
            error("Unexpected signature in @primitive: $(sig)")
        end
    end
    tsig = :($fn($(targs...)))
    return tsig, fn, vnames
end


function _primitive(sig)
    @assert sig.head == :call
    p_sig, fn, vnames = primitive_sig(sig)
    ex = :($p_sig = record!($(vnames[1]).tape, $Call, $fn, ($(vnames...),)))
    return ex
end

macro primitive(sig)
    return esc(_primitive(sig))
end
