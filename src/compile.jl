const NEXT_UNIQUE_ID = Ref{Int}(0)
next_unique_id() = (NEXT_UNIQUE_ID[] += 1; NEXT_UNIQUE_ID[])

make_name(id::Int, prefix="") = Symbol("$(prefix)x$id")
make_name(op::AbstractOp, prefix="") = Symbol("$(prefix)x$(op.id)")
make_name(name::String, prefix="") = Symbol("$(prefix)$(name)")

arg2expr(v::Variable, prefix="") = make_name(v.id, prefix)
arg2expr(s::Symbol, prefix="") = QuoteNode(s)
arg2expr(c, prefix="") = c

function to_expr(op::Call, prefix="")
    call = Expr(:call, [arg2expr(v, prefix) for v in (op.fn, op.args...)]...)
    return Expr(:(=), make_name(op.id, prefix), call)
end

to_expr(op::Constant, prefix="") = :($(make_name(op.id, prefix)) = $(op.val))


function loop_exit_tuple_expr_at_point(op::Loop, id::Int, prefix::String, loop_prefix::String)
    exit_name = make_name(op.id, prefix)
    arg_vars = loop_exit_vars_at_point(op, id)
    arg_names = [make_name(v.id, loop_prefix) for v in arg_vars]
    return Expr(:(=), exit_name, Expr(:call, tuple, arg_names...))
end


function to_expr(op::Loop, prefix="")
    loop_prefix = "l$(next_unique_id())"
    exprs = []
    # map parent input ids to continue ids
    init_var_names = []
    for (inp, parent) in zip(inputs(op.subtape), op.parent_inputs)
        init_var_name = make_name(inp.id, loop_prefix)
        push!(init_var_names, init_var_name)
        ex = Expr(:(=), init_var_name, make_name(parent.id, prefix))
        push!(exprs, ex)
    end
    # add exit tuple which will be used in case of zero trip count
    init_exit_tuple_ex = loop_exit_tuple_expr_at_point(op, 0, prefix, loop_prefix)
    push!(exprs, init_exit_tuple_ex)
    loop_ex = :(while true end)
    body = loop_ex.args[2]
    for (id, subop) in enumerate(op.subtape)
        if !isa(subop, Input)
            subex = to_expr(subop, loop_prefix)
            if subex isa Vector
                push!(body.args, subex...)
            else
                push!(body.args, subex)
            end
            if id == op.condition.id
                exit_expr = :(if !$(make_name(op.condition.id, loop_prefix)) end)
                exit_body = exit_expr.args[2]
                # update exit tuple
                exit_tuple_ex = loop_exit_tuple_expr_at_point(op, id, prefix, loop_prefix)
                push!(exit_body.args, exit_tuple_ex)
                push!(exit_body.args, Expr(:break))
                push!(body.args, exit_expr)
            end
        end
    end
    # map continue vars to inputs
    for (inp, cont) in zip(inputs(op.subtape), op.cont_vars)
        ex = Expr(:(=), make_name(inp.id, loop_prefix), make_name(cont.id, loop_prefix))
        push!(body.args, ex)
    end
    push!(exprs, loop_ex)
end


function to_expr(tape::Tape)
    fn_name = gensym("grad_$(tape[V(1)].val)")
    header = Expr(:call, fn_name)
    for v in inputs(tape)
        op = tape[v]
        push!(header.args, Expr(:(::), make_name(op), op.typ))
    end
    body = Expr(:block)
    for op in tape
        op isa Input && continue
        ex = to_expr(op)
        if ex isa Vector
            push!(body.args, ex...)
        else
            push!(body.args, ex)
        end
    end
    push!(body.args, Expr(:return, make_name(tape.result.id)))
    fn_ex = Expr(:function, header, body)
    return fn_ex
end


compile(tape::Tape) = Base.eval(@__MODULE__, to_expr(tape))