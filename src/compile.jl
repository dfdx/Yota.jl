## tape compilation

## CONVERSION TO EXGRAPH

make_name(op::AbstractOp) = Symbol("%$(getid(getvar(op)))")
make_name(var::TAny) = Symbol("%$(getid(var))")

to_exnode(op::Input) = ExNode{:input}(make_name(op), make_name(op); val=getvalue(op))
to_exnode(op::Constant) = ExNode{:constant}(make_name(op), getvalue(op); val=getvalue(op))
to_exnode(op::Assign) = ExNode{:(=)}(make_name(op), make_name(op.src); val=getvalue(op))


function to_exnode(op::Call)
    arg_names = map(make_name, op.args)
    # fns = maybe_to_symbol(op.fn)
    if isempty(op.kwargs)
        ex = Expr(:call, op.fn, arg_names...)
    else
        ex = Expr(:call, op.fn, Espresso.make_kw_params(op.kwargs), arg_names...)
    end
    return ExNode{:call}(make_name(op), ex; val=getvalue(op))
end

function to_exnode(op::Bcast)
    arg_names = map(make_name, op.args)
    #  fns = maybe_to_symbol(op.fn)
    ex = Expr(:., op.fn, Expr(:tuple, arg_names...))
    ExNode{:bcast}(make_name(op), ex; val=getvalue(op))
end


function to_exgraph(tape::Tape)
    g = ExGraph()
    for op in tape
        push!(g, to_exnode(op))
    end
    return g
end


## CODE GENERATION

"""
Rewrite :(Z = X * Y) into :(mul!(Z, X, Y)),  but only if X and Y are arrays
"""
function rewrite_mul(tape::Tape, ex::Expr)
    @assert ex.head == :block
    new_ex = Expr(:block)
    for subex in ex.args
        if (matchingex(:(_R = $*(_X, _Y)), subex)
            && tape[subex.args[1]].val isa AbstractArray
            && all(tape[subex.args[2].args[i]].val isa AbstractArray for i=2:3))
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
        push!(ex.args, :($name = $(op.var).val))
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
        push!(ex.args, :($(op.var).val = $name))
    end
    return ex
end


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


## COMPILATION

function compile(tape::Tape)
    fn_ex = generate_function_expr(tape)
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
        Base.invokelatest(tape.compiled)
    else
        for op in tape
            exec!(tape, op)
        end
    end
end
