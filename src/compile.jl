########################################################################
#                         CONVERSION TO EXGRAPH                        #
########################################################################

make_name(id::Int) = Symbol("%$id")
make_name(op::AbstractOp) = Symbol("%$(op.id)")

unmake_name(x::Symbol) = parse(Int, string(x)[2:end])

to_exnode(op::Input) = ExNode{:input}(make_name(op), make_name(op); val=op.val)
to_exnode(op::Assign) = ExNode{:(=)}(make_name(op), make_name(op.src_id); val=op.val)

function to_exnode(op::Constant)
    val = op.val isa Symbol ? QuoteNode(op.val) : op.val
    return ExNode{:constant}(make_name(op), val; val=val)
end


function to_exnode(op::Call)
    arg_names = map(make_name, op.args)
    ex = Expr(:call, op.fn, arg_names...)
    return ExNode{:call}(make_name(op), ex; val=op.val)
end


function to_exgraph(tape::Tape)
    g = ExGraph()
    for op in tape
        push!(g, to_exnode(op))
    end
    return g
end


########################################################################
#                       CODE GENERATION                                #
########################################################################

"""
Rewrite :(Z = X * Y) into :(mul!(Z, X, Y)), but only if X and Y are arrays
"""
function rewrite_mul(tape::Tape, ex::Expr)
    @assert ex.head == :block
    new_ex = Expr(:block)
    for subex in ex.args
        if (matchingex(:(_R = $*(_X, _Y)), subex)
            && tape[unmake_name(subex.args[1])].val isa AbstractArray
            && all(tape[unmake_name(subex.args[2].args[i])].val isa AbstractArray for i=2:3))
            new_subex = rewrite(subex, :(_Z = $*(_X, _Y)), :($mul!(_Z, _X, _Y)))
            push!(new_ex.args, new_subex)
        else
            push!(new_ex.args, subex)
        end
    end
    return new_ex
end


"""
Generate initialization block of function code
"""
function generate_prologue(tape::Tape)
    ex = Expr(:body)
    for op in tape
        name = make_name(op)
        push!(ex.args, :($name = $(op).val))
    end
    return ex
end


"""
Main part of generated function code
"""
function generate_body(tape::Tape)
    ret_var_ids = vcat(values(tape.derivs) |> collect, [tape.resultid])
    ret_var_names = [Symbol("%$id") for id in ret_var_ids]
    exg = to_exgraph(tape)
    # exg = Espresso.fuse_broadcasting(exg)  # TODO: this breaks test_examples.jl
    exg = Espresso.fuse_assigned(exg; outvars=ret_var_names)
    exg = Espresso.eliminate_common(exg)
    ex = Espresso.to_expr_kw(exg)
    # ex = rewrite_all(ex, INPLACE_RULES; phs=Set([:X, :Y, :Z]))
    ex = rewrite_mul(tape, ex)
    return ex
end


"""
Generate last block of function code
"""
function generate_epilogue(tape::Tape)
    ex = Expr(:body)
    for op in tape
        name = make_name(op)
        push!(ex.args, :($(op).val = $name))
    end
    return ex
end


"""
Generate function expression from the tape, binding all variables to tape's buffers
and optimizing code.
"""
function generate_function_expr(tape::Tape)
    fn_ex = :(function $(gensym("tape_fn"))() end)
    fn_ex_body = fn_ex.args[2]
    prologue = generate_prologue(tape)
    body = generate_body(tape)
    epilogue = generate_epilogue(tape)
    # abusing LineNumberNode to insert comments
    push!(fn_ex_body.args, LineNumberNode(0, "prologue"))
    push!(fn_ex_body.args, prologue.args...)
    push!(fn_ex_body.args, LineNumberNode(0, "body"))
    push!(fn_ex_body.args, body.args...)
    push!(fn_ex_body.args, LineNumberNode(0, "epilogue"))
    push!(fn_ex_body.args, epilogue.args...)
    return fn_ex
end


"""
Generate function expression from the tape without any optimizations or binding to
tape's buffers.

`ret_grad` optional parameter controls whether function should return only
value of tape[resultid] (default) or also values of gradient nodes
"""
function generate_function_expr_unbound(tape::Tape; ret_grad=false)
    # fn_args = [Expr(:(::), make_name(inp), typeof(inp.val)) for inp in tape if isa(inp, Input)]
    fn_args = [make_name(inp) for inp in tape if isa(inp, Input)]
    fn_ex = :(function $(gensym("tape_fn"))($(fn_args...)) end)
    fn_ex_body = fn_ex.args[2]
    for op in tape
        if !isa(op, Input)
            ex = op |> to_exnode |> to_expr
            push!(fn_ex_body.args, ex)
        end
    end
    if ret_grad
        res_var = make_name(tape[tape.resultid])
        grad_vars = []
        for op in tape
            if op isa Input
                if haskey(tape.derivs, op.id)
                    push!(grad_vars, make_name(tape.derivs[op.id]))
                else
                    push!(grad_vars, nothing)
                end
            end
        end
        ret_tuple = Expr(:tuple, res_var, grad_vars...)
        push!(fn_ex_body.args, :(return $ret_tuple))
    else
        push!(fn_ex_body.args, :(return $(make_name(tape[tape.resultid]))))
    end
    return fn_ex
end


########################################################################
#                            COMPILATION                               #
########################################################################


function compile(tape::Tape; bind=true, ret_grad=false)
    fn_ex = bind ? generate_function_expr(tape) : generate_function_expr_unbound(tape; ret_grad=ret_grad)
    return Core.eval(@__MODULE__, fn_ex)
end


function compile!(tape::Tape)
    tape.compiled = compile(tape)
end
