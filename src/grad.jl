
# TODO: track_and_record!(tape, op)??

function back!(tape::Tape)
    # z - final variable, y - resulting variable of current op, x - dependencies of y
    # dy - derivative of z w.r.t. y
    z = tape[end].var
    dy = tracked(tape, 1.0)
    record!(tape, Constant(dy))
    # set initial derivative value
    tape.derivs[z.id] = dy.id
    for op in reverse(tape.ops[1:end-1])
        if op isa Call
            for i=1:length(op.args)
                rev_step!(op, i)
            end
        end
    end
end


getderiv(tape::Tape, id::Int) = tape[tape.derivs[id]].var
getderiv(tape::Tape, var::TAny) = getderiv(tape, var.id)
setderiv!(tape::Tape, var_id::Int, grad_var_id::Int) = (tape.derivs[var_id] = grad_var_id)
setderiv!(tape::Tape, var::TAny, grad_var::TAny) = (tape.derivs[var.id] = grad_var.id)


# @inline rev_step!(op::Input, i::Int) = ()

function rev_step!(op::Call, i::Int)
    tape = op.var.tape
    y = op.var
    x = op.args[i]
    dy = getderiv(tape, y)
    dx = grad!(dy, Val(i), op)
    if !haskey(tape.derivs, x.id)
        setderiv!(tape, x, dx)
    else
        old_dx = getderiv(tape, x)
        new_dx = record!(tape, Call(zero(dx), +, (dx, old_dx)))
        setderiv!(tape, x, new_dx)
    end
end




function grad!(dy::TAny, ::Val{1}, op::Call{typeof(*), Tuple{TReal, TReal}})
    x, y = op.args
    grad_var = zero(op.args[2])
    return record!(dy.tape, Call(grad_var, *, (dy, y)))
end

function grad!(dy::TAny, ::Val{2}, op::Call{typeof(*), Tuple{TReal, TReal}})
    x, y = op.args
    grad_var = zero(op.args[2])
    return record!(dy.tape, Call(grad_var, *, (dy, x)))
end
