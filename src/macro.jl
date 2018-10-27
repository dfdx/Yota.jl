
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


## @grad

function grad_sig(sig, idx)
    fn, args = sig.args[1], sig.args[2:end]
    targs = Array{Any}(undef, length(args))
    type_params = Symbol[]
    type_param_idx = 1
    for i=1:length(args)
        if args[i] isa Expr
            name, typ_ex = args[i].args
            typ = Core.eval(Base, typ_ex)
            if typ  <: AbstractArray
                T, N = Symbol("T$type_param_idx"), Symbol("N$type_param_idx")
                type_param_idx += 1
                push!(type_params, T, N)
                ttyp = Expr(:curly, TArray, T, N)
            else
                ttyp = TReal
            end
            targs[i] = ttyp
        elseif args[i] isa Symbol
            name = args[i]
            targs[i] = TAny
        else
            error("Unexpected signature in @primitive: $(sig)")
        end
    end
    grad_fn = Expr(:., Symbol(@__MODULE__), QuoteNode(:grad!))
    if isempty(type_params)
        tsig = :($grad_fn(dy::$TAny, ::Val{$idx}, op::$Call{typeof($fn), Tuple{$(targs...)}}))
    else
        tsig = :($grad_fn(dy::$TAny, ::Val{$idx}, op::$Call{typeof($fn), Tuple{$(targs...)}})
                 where {$(type_params...)})
    end
    return tsig
end


function _grad(sig, idx, body)
    tsig = grad_sig(sig, idx)
    ex = :($tsig = $body)
    return ex
end

macro grad(sig, idx, body)
    esc(_grad(sig, idx, body))
end
