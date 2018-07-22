
function grad!(dy::TAny, ::Val{1},
               op::Call{typeof(*), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    x, y = op.args
    yt = record!(dy.tape, Call, transpose, (y,))
    return record!(dy.tape, Call, *, (dy, yt))
end

function grad!(dy::TAny, ::Val{2},
               op::Call{typeof(*), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    x, y = op.args
    xt = record!(dy.tape, Call, transpose, (x,))
    return record!(dy.tape, Call, *, (xt, dy))
end


function grad!(dy::TAny, ::Val{1}, op::Call{typeof(sum), Tuple{TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Call, sum_grad, (op.args[1], dy); kwargs=op.kwargs)
end
function grad!(dy::TAny, ::Val{1}, op::Call{typeof(mean), Tuple{TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Call, sum_grad, (op.args[1], dy); kwargs=op.kwargs)
end


grad!(dy::TAny, ::Val{1}, op::Call{typeof(transpose), Tuple{TArray{T,N}}}) where {T,N} =
    transpose(dy)
# TODO: .==
# TODO: minimum, maximum



## BROADCASTING

getfirst(x::TArray) = first(x.val)
getfirst(x) = first(x)

# function record_bcast_as_calls(tape::Tape)
#     new_tape = Tape()
#     for op in tape.ops
#         if op isa Call
#             bcast_op = Bcast(op.var, op.fn, op.args)
#             _record!(new_tape, bcast_op)
#         else
#             _record!(new_tape, op)
#         end
#     end
#     return new_tape
# end


# function calls_to_bcast(tape::Tape)
#     new_tape = Tape()
#     for op in tape.ops
#         if op isa Call
#             bcast_op = Bcast(op.var, op.fn, op.args)
#             _record!(new_tape, bcast_op)
#         else
#             _record!(new_tape, op)
#         end
#     end
#     return new_tape
# end

# function replace_var!(tape::Tape, from::TAny, to::Any)
#     for op in tape.ops
#         println(op)
#         if op isa Call || op isa Bcast
#             if op.var === from
#                 op.var = to
#             end
#             op.args = ([arg === from ? to : arg for arg in op.args]...,)
#         end
#     end
# end

# function replace_vars!(tape::Tape, from_to)
#     for (from, to) in from_to
#         replace_var(tape, from, to)
#     end
# end


function merge_bcast_as_call!(tape::Tape, minitape::Tape, collect_from::Int, var_st::Dict)
    for i=collect_from:length(minitape)
        op = minitape[i]
        if op isa Call
            tape_args = map(x -> tape[var_st[x.id]].var, op.args)
            var = record!(tape, Bcast, op.fn, tape_args)
        elseif op isa Constant
            # TODO: replace var and args in op if it makes sense for the op
            var = record!(tape, Constant, op.var.val)
        elseif op isa Assign
            var = record!(tape, Assign, var_st[op.var.id])
        else
            error("Unexpected operation in gradient minitape: $op")
        end
        var_st[op.var.id] = var.id  # add to mapping of minitape IDs to tape IDs
    end
end

function grad!(dy::Any, ::Val{Idx},
               op::Bcast{FnType, Tuple{TArray{T,N}, TArray{T,N}}}) where {Idx,FnType,T,N}
    # record derivatives of first elements onto a minitape
    minitape = Tape()
    el_args = map(arg -> record!(minitape, Input, getfirst(arg)), op.args)
    # run grad! for first elements on the minitape
    el_y = op.fn(el_args...)
    el_dy = record!(minitape, Constant, getfirst(dy))
    el_op = minitape[el_y.id]
    collect_from = length(minitape) + 1
    el_dx = grad!(el_dy, Val(Idx), el_op)
    var_st = Dict(el_a.id => a.id for (el_a, a) in zip(el_args, op.args))
    var_st[el_dy.id] = y.id
    merge_bcast_as_call!(tape, minitape, collect_from, var_st)
    dx = tape[var_st[el_dx.id]].var
    return dx
end


# TODO: make single function to handle all broadcasting of arrays of the same size
function grad!(dy::TAny, ::Val{1}, op::Bcast{typeof(+), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Assign, dy)
end

function grad!(dy::TAny, ::Val{2}, op::Bcast{typeof(+), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
    return record!(dy.tape, Assign, dy)
end
