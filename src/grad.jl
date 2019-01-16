########################################################################
#                            GRAD RESULT                               #
########################################################################

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

function Base.getindex(g::GradResult, argid::Int)
    tape = g.tape
    gvar = g.gvars[argid]
    if isa(gvar, Dict)
        return Dict(f => tape[id].var.val for (f, id) in gvar)
    else
        return tape[gvar].var.val
    end
end


########################################################################
#                              GRAD                                    #
########################################################################

getderiv(tape::Tape, id::Int) = tape[tape.derivs[id]]
getderiv(tape::Tape, op::AbstractOp) = getderiv(tape, op.id)
setderiv!(tape::Tape, op_id::Int, grad_op_id::Int) = (tape.derivs[op_id] = grad_op_id)
setderiv!(tape::Tape, op::AbstractOp, grad_op::AbstractOp) = (tape.derivs[op.id] = grad_op.id)


Espresso.to_expr(tape::Tape, op::Call) = begin
    @assert isempty(op.kwargs) "Oops, functions with kwargs aren't supported just yet"
    Expr(:call, op.fn, [Symbol("%$i") for i in op.args]...)
end

to_unbroadcast_expr(tape::Tape, op::Call) =
    Expr(:call, tape[op.args[1]].val, [Symbol("%$i") for i in op.args[2:end]]...)


function deriv!(tape::Tape, op::AbstractOp, i::Int, dy::AbstractOp)
    ex = to_expr(tape, op)
    dep_types = [tape[arg].typ for arg in op.args]
    dex = deriv_expr(ex, dep_types, i)
    st = Dict(Symbol("%$i") => i for i in op.args)
    st[:ds] = dy.id
    ret_id = record_expr!(tape, dex; st=st)
    return tape[ret_id]
end


function deriv_broadcast!(tape::Tape, op::AbstractOp, i::Int, dy::AbstractOp)
    # 1. take basic elements (see in Yota, presumably just first())
    # 2. find rule for basic elements
    # 3. record_expr_broadcast!() - record `broadcast` and execute immediately
    ex = to_unbroadcast_expr(tape, op)
    dep_eltypes = [eltype(tape[arg].typ) for arg in op.args[2:end]]
    dex = deriv_expr(ex, dep_eltypes, i-1)
    st = Dict(Symbol("%$id") => id for id in op.args)
    st[:ds] = dy.id
    ret_id = record_expr!(tape, dex; st=st, bcast=true)
    return tape[ret_id]
end


function step_back!(tape::Tape, op::Union{Call}, i::Int)
    y = op
    x = tape[op.args[i]]
    dy = getderiv(tape, y)
    dx = (op.fn == broadcast ?
          deriv_broadcast!(tape, op, i, dy) :
          deriv!(tape, op, i, dy))
    if !haskey(tape.derivs, x.id)
        setderiv!(tape, x, dx)
    else
        @warn "This branch hasn't been tested yet"
        old_dx = getderiv(tape, x)
        val = dx.val + old_dx.val
        new_dx_id = record!(tape, Call, val, +, [dx.id, old_dx.id])
        new_dx = tape[new_dx_id]
        setderiv!(tape, x, new_dx)
    end
end


function back!(tape::Tape)
    # z - final variable (usually a loss)
    # y - resulting variable of current op
    # x - dependencies of y
    # dy - derivative of z w.r.t. y
    z = tape[tape.resultid]
    # using Float32 for seed since for 64-bit args it will be expanded anyway
    dy_id = record!(tape, Constant, 1.0f0)
    dy = tape[dy_id]
    # set initial derivative value
    tape.derivs[z.id] = dy.id
    for op in reverse(tape.ops[1:end-1])
        println(op)
        if op isa Call # || op isa Bcast
            for i=1:length(op.args)
                # backpropagate only non-constant vars
                # note that it also prevents backprop on 1st param of broadcast
                arg_op = tape[op.args[i]]
                if !isa(arg_op, Constant)
                    step_back!(tape, op, i)
                end
            end
        end
    end
end



"""
For each input that has a derivative on this tape check if the derivative
has the same size as the input.
"""
function check_deriv_sizes(tape::Tape)
    for (var_id, grad_var_id) in tape.derivs
        var_size = size(tape[var_id].val)
        grad_var_size = size(tape[grad_var_id].val)
        if  var_size != grad_var_size
            @warn "Gradient %$grad_var_id has size $grad_var_size, " *
                "but original variable %$var_id has size $var_size"
        end
    end
end


function _grad(f::Function, args...)
    val, tape = trace(f, args...)
    tape = simplify(tape)
    # backpropagate gradients
    back!(tape)
    # consistency check
    check_deriv_sizes(tape)
    # construct GradResult object that wraps tape and provide accessors for computed derivatives
    return val, GradResult(tape)
end
