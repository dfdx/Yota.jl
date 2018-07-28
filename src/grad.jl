## gradient function defintions

## GRAD RESULT

struct GradResult
    tape::Tape
    gvars::Dict{Int, Any}  # gradient vars: argid -> gradient var
end


function GradResult(tape::Tape)
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
    return GradResult(tape, gvars)
end


Base.show(io::IO, g::GradResult) = print(io, "GradResult($(length(g.gvars)))")

function getindex(g::GradResult, argid::Int)
    tape = g.tape
    gvar = g.gvars[argid]
    if isa(gvar, Dict)
        return Dict(f => tape[id].var.val for (f, id) in gvar)
    else
        return tape[gvar].var.val
    end
end


## GRAD

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
                # println("op = $op; i = $i")
                # backpropagate only non-constant tracked vars
                arg_op = tape[getid(op.args[i])]
                if op.args[i] isa TAny && !isa(arg_op, Constant)
                    rev_step!(op, i)
                end
            end
        end
    end
end


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


function _grad(f::Function, args...)
    tape = Tape()
    # wrap args into tracked data
    targs = make_tracked_args(tape, args...)
    # execute function to fill in the tape
    tres = f(targs...)
    tape.resultid = getid(tres)
    # backpropagate gradients
    back!(tape)
    # construct GradResult object that wraps tape and provide accessors for computed derivatives
    return tres.val, GradResult(tape)
end


const GRAD_CACHE = Dict{Any, Tape}()

function grad(f::Function, args...; static=true)
    if static
        # key conists of function type and type of argument (for structs) or its size
        cache_key = (f, ([isstruct(arg) ? typeof(arg) : size(arg) for arg in args]...,))
        if haskey(GRAD_CACHE, f)
            tape = GRAD_CACHE[cache_key]
            play!(tape, args...)
            return getvalue(tape[tape.resultid]), GradResult(tape)
        else
            val, g = _grad(f, args...)
            compile!(g.tape)
            GRAD_CACHE[cache_key] = g.tape
            return val, g
        end
    else
        return _grad(f, args...)
    end
end
