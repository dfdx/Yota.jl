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


function make_tracked_args(tape::Tape, args...)
    targs = []
    for (argid, arg) in enumerate(args)
        if isstruct(arg)
            # we'd like to avoid deepcopy, but it's not clear yet how to make shallow one
            # note: shallow copy will also require changes in record_struct!
            arg = deepcopy(arg)
            record_struct!(tape, arg, argid)
            targ = arg  # we use the same (copy of) struct, but all fields are rewritten
        else
            targ = record!(tape, Input, arg; argid=argid)
        end
        push!(targs, targ)
    end
    return targs
end



function grad(f::Function, args...)
    tape = Tape()
    # wrap args into tracked data
    targs = make_tracked_args(tape, args...)
    # execute function to fill in the tape
    tres = f(targs...)
    res_id = tape[end].var
    # backpropagate gradients
    back!(tape)
    # construct Grads object that wraps tape and provide accessors for computed derivatives
    return tres.val, Grads(tape)
end



struct Grads
    tape::Tape
    # gradient vars: argid -> gradient var
    gvars::Dict{Int, Any}
end


function Grads(tape::Tape)
    gvars = Dict{Int,Any}()
    # struct fields
    for (argid, dct) in tape.sfields
        gvars[argid] = Dict(field_path => tape.derivs[var_id]
                            for (field_path, var_id) in dct
                            if haskey(tape.derivs, var_id))  # not all fields may have derivatives
    end
    # other arguments
    struct_arg_ids = Set(keys(tape.sfields))
    for op in tape
        if op isa Input && !in(op.argid, struct_arg_ids)
            gvars[op.argid] = tape.derivs[op.var.id]
        end
    end
    return Grads(tape, gvars)
end


Base.show(io::IO, g::Grads) = print(io, "Grads($(length(g.gvars)))")

function getindex(g::Grads, argid::Int)
    tape = g.tape
    gvar = g.gvars[argid]
    if isa(gvar, Dict)
        return Dict(f => tape[id].var.val for (f, id) in gvar)
    else
        return tape[gvar].var.val
    end
end



## TAPE COMPILATION

function to_expr(op::Call)
    if op.var.val isa AbstractArray
        return :($(op.var).val .= $(op.fn)(map(getvalue, $(op.args))...))
    else 
        return :($(op.var).val = $(op.fn)(map(getvalue, $(op.args))...))
    end
end
function to_expr(op::Call{typeof(*), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    return :(mul!($(op.var).val, $(op.args[1]).val, $(op.args[2]).val))
end
to_expr(op::Bcast) = :($(op.var).val .= $(op.fn).(map(getvalue, $(op.args))...))
to_expr(op::Assign) = :($(op.var).val = $(op.var).src)
to_expr(op::Constant) = :()    # constants don't change anyway


function compile(tape::Tape)
    fn_ex = :(function $(gensym("tape_fn"))() end)
    # do we need to pass inputs as args? it seems like we can do it manually before invoking
    # the cached function and let it just use pre-existing vars
    # head = fn_ex.args[1]
    body = fn_ex.args[2]
    for op in tape
        if !isa(op, Input)
            ex = to_expr(op)
            push!(body.args, ex)
        end
    end
    return Core.eval(@__MODULE__, fn_ex)
end


function compile!(tape::Tape)
    tape.compiled = compile(tape)
end


function rerecord_inputs!(tape::Tape, args...)
    minitape = Tape()
    targs = make_tracked_args(minitape, args...)
    for i=1:length(minitape)
        val = getvalue(minitape[i])
        setvalue!(tape[i], val)
    end
end


function play!(tape::Tape, args...; use_compiled=true)
    rerecord_inputs!(tape, args...)
    if use_compiled && tape.compiled != nothing
        tape.compiled()
    else        
        for op in tape
            exec!(tape, op)
        end
    end
end
