import Statistics
using LinearAlgebra
using OrderedCollections
using ChainRulesCore
using ChainRules
using NNlib
using Ghost
using Ghost: Tape, Variable, V, Call, mkcall, Constant, inputs
using Ghost: bound, compile, play!, isstruct
using Ghost: remove_first_parameter, kwfunc_signature, call_signature


include("helpers.jl")
include("drules.jl")
include("chainrules.jl")
include("grad.jl")
include("update.jl")
include("gradcheck.jl")


function __init__()
    update_chainrules_primitives!()
end


# TODO: move to Ghost
function show_compact(tape::Tape)
    println(typeof(tape))
    dont_show = Set([])
    for op in tape
        if in(op.id, dont_show)
            continue
        elseif op isa Call && op.val isa Tuple
            out_vars = []
            destructured = false
            for i=1:length(op.val)
                var = "_"
                for id=(op.id + 1:length(tape))
                    opc = tape[V(id)]
                    if opc isa Call && opc.fn in (getfield, _getfield) &&
                        opc.args[1].id == op.id && opc.args[2] == i
                        var = "%$(opc.id)"
                        push!(dont_show, opc.id)
                        destructured = true
                        break
                    end
                end
                push!(out_vars, var)
            end
            if destructured
                out_vars_str = join(out_vars, ", ")
                arg_str = join(op.args, ", ")
                println("  $out_vars_str = $(op.fn)($arg_str)")
            else
                println("  $op")
            end
        else
            println("  $op")
        end
    end
end
