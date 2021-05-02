const DRULES = Ref(FunctionResolver{Function}())


function expr2signature(modl::Module, ex::Expr)
    BAD_FORMAT_MSG = "Expected signature as `fn(x::T1, y::T2)`"
    @assert Meta.isexpr(ex, :call) BAD_FORMAT_MSG
    fn_typ = getfield(modl, ex.args[1]) |> typeof
    arg_typs = []
    for x in ex.args[2:end]
        @assert Meta.isexpr(x, :(::)) BAD_FORMAT_MSG
        # Module.eval() is slower, but supports UnionAll
        typ = modl.eval(x.args[end])
        push!(arg_typs, typ)
    end
    return Tuple{fn_typ, arg_typs...}
end


macro drule(ex, df)
    let sig = expr2signature(@__MODULE__, ex)
        quote
            $DRULES[][$sig] = $df
            nothing
        end
    end
end


get_deriv_function(sig) = DRULES[][sig]

is_yota_primitive(sig) = sig in DRULES[]


###############################################################################
#                                    RULES                                    #
###############################################################################

function ∇multiply(dy, ::typeof(*), x::Number, y::Number)
    return NO_FIELDS, dy * y, dy * x
end

@drule *(x::Number, y::Number) ∇multiply