## tape compilation

function to_expr(op::Call)
    if op.var.val isa AbstractArray
        return :($(op.var).val .= $(op.fn)(map(getvalue, $(op.args))...; $(op.kwargs)...))
    else
        return :($(op.var).val = $(op.fn)(map(getvalue, $(op.args))...; $(op.kwargs)...))
    end
end
function to_expr(op::Call{typeof(*), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    return :(mul!($(op.var).val, $(op.args[1]).val, $(op.args[2]).val))
end

function to_expr(op::Bcast)
    if op.var.val isa AbstractArray
        return :($(op.var).val .= $(op.fn).(map(getvalue, $(op.args))...))
    else
        return :($(op.var).val = $(op.fn).(map(getvalue, $(op.args))...))
    end
end
# to_expr(op::Bcast) = :($(op.var).val .= $(op.fn).(map(getvalue, $(op.args))...))
to_expr(op::Assign) = :($(op.var).val = $(op.src).val)
to_expr(op::Constant) = :()    # constants don't change anyway


function compile(tape::Tape)
    fn_ex = :(function $(gensym("tape_fn"))() end)
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
        tape.compiled()  # TODO: Base.invokelatest?
    else
        for op in tape
            exec!(tape, op)
        end
    end
end
