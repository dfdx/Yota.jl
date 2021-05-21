make_name(id::Int) = Symbol("x$id")
make_name(op::AbstractOp) = Symbol("x$(op.id)")

arg2expr(v::Variable) = make_name(v.id)
arg2expr(s::Symbol) = QuoteNode(s)
arg2expr(c) = c

function to_expr(op::Call)
    call = Expr(:call, map(arg2expr, (op.fn, op.args...))...)
    return Expr(:(=), make_name(op.id), call)
end

to_expr(op::Constant) = :($(make_name(op.id)) = $(op.val))


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
        push!(body.args, to_expr(op))
    end
    push!(body.args, Expr(:return, make_name(tape.result.id)))
    fn_ex = Expr(:function, header, body)
    return fn_ex
end


compile(tape::Tape) = Base.eval(@__MODULE__, to_expr(tape))