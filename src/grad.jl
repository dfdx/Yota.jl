
# TODO: track_and_record!(tape, op)??

function reverse_pass!(tape::Tape)
    # z - final variable, y - resulting variable of current op, x - dependencies of y
    # dy - derivative of z w.r.t. y
    z = tape[end].var
    dy = tracked(tape, 1.0)
    record!(tape, Constant(dy))
    for op in reverse(tape.ops[1:end-1])
        for i=1:length(op.args)
            rev_step!(op, i)  # dy - ???
        end
    end
end


function rev_step!(op::AbstractOp, i::Int)
    tape = op.var.tape
    dy = tape.derivs[op.var.id]
    grad!(dy, Val(i), op)
end




getdata(x::TReal) = x.data
getdata(x) = x




function grad!(dy::TAny, ::Val{1}, op::Call{typeof(*), Tuple{TReal, TReal}})
    x, y = op.args
    grad_var = record!(dy.tape, Call(tracked(dy.tape, zero(x.data)), *, (dy, y)))
    record!(dy.tape, AddGrad(x, grad_var))
end
