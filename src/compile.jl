## tape compilation

## epoch 1

# function to_expr(op::Call)
#     if op.var.val isa AbstractArray
#         return :($(op.var).val .= $(op.fn)(map(getvalue, $(op.args))...; $(op.kwargs)...))
#     else
#         return :($(op.var).val = $(op.fn)(map(getvalue, $(op.args))...; $(op.kwargs)...))
#     end
# end
# function to_expr(op::Call{typeof(*), Tuple{TArray{T,N}, TArray{T,N}}}) where {T,N}
#     return :(mul!($(op.var).val, $(op.args[1]).val, $(op.args[2]).val))
# end

# function to_expr(op::Bcast)
#     if op.var.val isa AbstractArray
#         return :($(op.var).val .= $(op.fn).(map(getvalue, $(op.args))...))
#     else
#         return :($(op.var).val = $(op.fn).(map(getvalue, $(op.args))...))
#     end
# end
# # to_expr(op::Bcast) = :($(op.var).val .= $(op.fn).(map(getvalue, $(op.args))...))
# to_expr(op::Assign) = :($(op.var).val = $(op.src).val)


# epoch 2

# make_name(op::AbstractOp) = Symbol("%$(getid(getvar(op)))")
# make_name(var::TAny) = Symbol("%$(getid(var))")


# function to_expr(op::Call)
#     arg_names = map(make_name, op.args)
#     if isempty(op.kwargs)
#         rhs = Expr(:call, op.fn, arg_names...)
#     else
#         rhs = Expr(:call, op.fn, Espresso.make_kw_params(op.kwargs), arg_names...)
#     end
#     return :($(make_name(op)) = $rhs)
# end

# function to_expr(op::Bcast)
#     arg_names = map(make_name, op.args)
#     return :($(make_name(op)) = $(op.fn).($(arg_names...)))
# end

# to_expr(op::Assign) = :($(make_name(op)) = $(make_name(op.src)))
# to_expr(op::Constant) = :($(make_name(op)) = $(getvalue(op)))

# function to_expr(tape::Tape)
#     # TODO: add init block, i.e. %n = $(tape[n].var.val)
#     args = []
#     for op in tape
#         if !isa(op, Input)
#             ex = to_expr(op)
#             push!(args, ex)
#         end
#     end
#     # push!(args, LineNumberNode(0, "comment"))
#     # TODO: add finale block, i.e. $(tape[n].var.val) = %n
#     # TODO: define output vars and do all the fun stuff from Espresso codegen
#     return Expr(:block, args...)
#     # TODO: maybe use LineNumberNodes as comments in AST, e.g. LineNumberNode(0, "comment")
#     # TODO: it might be ok to use Symbol(op.fn) instead of op.fn since it returns qualified name
#     #       although it might fail with Yota/Main functions
#     #       anyway, op.fn should work fine too
# end


## epoch 3

## CONVERSION TO EXGRAPH

make_name(op::AbstractOp) = Symbol("%$(getid(getvar(op)))")
make_name(var::TAny) = Symbol("%$(getid(var))")

to_exnode(op::Input) = ExNode{:input}(make_name(op), make_name(op); val=getvalue(op))
to_exnode(op::Constant) = ExNode{:constant}(make_name(op), getvalue(op); val=getvalue(op))
to_exnode(op::Assign) = ExNode{:(=)}(make_name(op), make_name(op.src); val=getvalue(op))

# const OK_SYMBOL_FUNCS = Set([*, ^, exp, log])

# Espresso operates on symbols, not function objects.
# Although it's fragile for many functions, it's useful for a number of standard functions.
# Later it lets us use a number of useful code transformations during code generation,
# specifically convert * to mul! and rewrite some functions to CUDAnative
# without adding it to dependencies
# An alternative approach would be to reimplement Espresso's optimizations
# for function objects instead of function names, leaving only CUDAnative stuff symbolic.
# maybe_to_symbol(f::Function) = f in OK_SYMBOL_FUNCS ? Symbol(f) : f


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

const INPLACE_RULES = [
    :(Z = $*($transpose(X), Y)) => :($mul!(Z, $transpose(X), Y)),
    :(Z = $*(X, $transpose(Y))) => :($mul!(Z, X, $transpose(Y))),
    :(Z = $*(X, Y)) => :($mul!(Z, X, Y)),
]


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
    # ex = to_expr(tape)
    # inputs = [Symbol("%$id") => getvalue(op) for (id, op) in enumerate(tape) if op isa Input]
    # exg = ExGraph(ex; inputs...)
    ret_var_ids = vcat(values(tape.derivs) |> collect, [tape.resultid])
    ret_var_names = [Symbol("%$id") for id in ret_var_ids]
    exg = to_exgraph(tape)
    exg = Espresso.fuse_broadcasting(exg)
    exg = Espresso.fuse_assigned(exg; outvars=ret_var_names)
    exg = Espresso.eliminate_common(exg)
    ex = Espresso.to_expr(exg)
    ex = rewrite_all(ex, INPLACE_RULES; phs=Set([:X, :Y, :Z]))
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
