## gradient function defintions

function back!(tape::Tape)
    # z - final variable, y - resulting variable of current op, x - dependencies of y
    # dy - derivative of z w.r.t. y
    z = tape[end].var
    dy = record!(tape, Constant, 1.0)
    # set initial derivative value
    tape.derivs[z.id] = dy.id
    for op in reverse(tape.ops[1:end-1])
        if op isa Call || op isa Bcast
            for i=1:length(op.args)
                # backpropagate only tracked vars
                if op.args[i] isa TAny
                    rev_step!(op, i)
                end
            end
        end
    end
end


getderiv(tape::Tape, id::Int) = tape[tape.derivs[id]].var
getderiv(tape::Tape, var::TAny) = getderiv(tape, var.id)
setderiv!(tape::Tape, var_id::Int, grad_var_id::Int) = (tape.derivs[var_id] = grad_var_id)
setderiv!(tape::Tape, var::TAny, grad_var::TAny) = (tape.derivs[var.id] = grad_var.id)


function rev_step!(op::Union{Call, Bcast}, i::Int)
    tape = op.var.tape
    y = op.var
    x = op.args[i]
    dy = getderiv(tape, y)
    dx = grad!(dy, Val(i), op)
    if !haskey(tape.derivs, x.id)
        setderiv!(tape, x, dx)
    else
        old_dx = getderiv(tape, x)
        new_dx = record!(tape, Call, +, (dx, old_dx))
        setderiv!(tape, x, new_dx)
    end
end




## high-level API


function grad(f::Function, args...)
    tape = Tape()
    # wrap args into tracked data
    targs = []
    for (argid, arg) in enumerate(args)
        if isstruct(arg)
            # we'd like to avoid deepcopy, but it's not clear yet how to make shallow one
            # note: shallow copy will also require changes in record_struct!
            arg = deepcopy(args)
            targ = record_struct!(tape, arg; argid=argid)
        else
            targ = record!(tape, Input, arg; argid=argid)
        end
        push!(targs, arg)
    end    
    # execute function to fill in the tape
    tres = f(targs...)
    # backpropagate gradients
    back!(tape)
    # return value and the tape
    return tres.val, tape
end



struct Grads
    # gradient vars: argid -> gradient var
    gvars::Dict{Int, Any}
end


function Grads(tape::Tape)
    for op in tape.ops
        if op isa Input
            # TODO
        end
    end
end

# 5. implement update! that can work with structs
# 6. way to exec! tape with new inputs
