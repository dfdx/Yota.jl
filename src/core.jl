import Statistics
using LinearAlgebra
using ChainRulesCore
using ChainRules
using NNlib
using Umlaut
# TODO: make these objects exported by default?
import Umlaut: Tape, Variable, V, Call, mkcall, Constant
import Umlaut: record_primitive!, isprimitive, BaseCtx
import Umlaut: play!, compile
import Ghost
import Ghost: bound, inputs


include("helpers.jl")
include("utils.jl")
include("deprecated.jl")
# include("drules.jl")
include("chainrules.jl")
include("rulesets.jl")
include("grad.jl")
include("update.jl")
include("gradcheck.jl")


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
                println("  $out_vars_str = [%$(op.id)] = $(op.fn)($arg_str)")
            else
                println("  $op")
            end
        else
            println("  $op")
        end
    end
end


Base.show(io::IO, tape::Tape{GradCtx}) = show_compact(tape)
