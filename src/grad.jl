function is_primitive(sig)
    return (Ghost.is_primitive(sig) ||
            is_yota_primitive(sig) ||
            is_chainrules_primitive(sig))
end


########################################################################
#                            GRAD CONTEXT                              #
########################################################################

struct GradCtx
    # map from primal var to its pullback var
    # note: LittleDict is required because vars are mutable
    pullbacks::LittleDict{Variable,Variable}
    # map from primal var to its derivative var
    derivs::LittleDict{Variable,Variable}
end

GradCtx() = GradCtx(LittleDict(), LittleDict())

function rebind_context!(tape::Tape{GradCtx}, st::Dict)
    for (v, dv) in tape.c.derivs
        rebind!(tape, v, st)
        rebind!(tape, dv, st)
    end
    for (v, dv) in tape.c.pullbacks
        rebind!(tape, v, st)
        rebind!(tape, dv, st)
    end
end

########################################################################
#                              GRAD                                    #
########################################################################

const DEBUG_STATE = Ref{Any}()


getderiv(tape::Tape, v::Variable) = get(tape.c.derivs, bound(tape, v), nothing)
setderiv!(tape::Tape, x::Variable, dx::Variable) = (
    tape.c.derivs[bound(tape, x)] = bound(tape, dx)
)
hasderiv(tape::Tape, v::Variable) = getderiv(tape, v) !== nothing


function set_or_add_deriv!(tape::Tape, x::Variable, dx::Variable)
    if !hasderiv(tape, x)
        setderiv!(tape, x, dx)
    else
        old_dx = getderiv(tape, x)
        if tape[dx].val isa Tangent || tape[old_dx].val isa Tangent
            # val = tape[dx].val + tape[old_dx].val
            # new_dx_id = push!(tape, Call, val, (+), [dx.id, old_dx.id])
            new_dx = push!(tape, mkcall(+, dx, old_dx))
        else
            # val = dx.val .+ old_dx.val
            # dot_add_id = record!(tape, Constant, +)
            # new_dx_id = record!(tape, Call, val, broadcast, [dot_add_id, dx.id, old_dx.id])
            new_dx = push!(tape, mkcall(broadcast, +, dx, old_dx))
        end
        setderiv!(tape, x, new_dx)
    end
end


# there's also Core.var"#isa##kw", but I don't understand its semantics
is_kwfunc(f) = endswith(string(f), "##kw")
is_kwfunc(v::Variable) = is_kwfunc(v._op.val)

"""
Transofrm the tape replacing calls `fn(args...)` for which `ChainRule.rrule` is defined
with the following chain or calls:

    * t = rrule(fn, args...)
    * val = _getfield(t, 1)
    * pb = _getfield(t, 2)

where `val = fn(args...)` and `pb` is the pullback function.
"""
function chainrules_transform!(tape::Tape)
    config = YotaRuleConfig()
    i = 1
    while i <= length(tape)
        op = tape[V(i)]
        if (op isa Call
                && is_chainrules_primitive(call_signature(tape, op))
                && !is_yota_primitive(call_signature(tape, op)))
            # replace f(args...) with rrule(f, args...)
            # if op.fn is a kw function, use kw version of rrule
            rr_op = (is_kwfunc(op.fn) ?
                    mkcall(Core.kwfunc(rrule), op.args[1], rrule, op.args[2:end]...) :
                    mkcall(rrule, config, op.fn, op.args...))
            @assert rr_op.val !== nothing "rrule($(op.fn), ...) returned nothing"
            val_op = mkcall(_getfield, V(rr_op), 1)
            pb_op = mkcall(_getfield, V(rr_op), 2)
            tape.c.pullbacks[V(val_op)] = V(pb_op)
            replace!(tape, i => [rr_op, val_op, pb_op]; rebind_to=2)
            i += 3  # rrule + 2 _getfield ops
        else
            i += 1
        end
    end
    return tape
end


"""
Collect variables that we need to step through during the reverse pass.
The returned vector is already deduplicated and reverse-sorted
"""
function todo_list(tape::Tape{GradCtx}, y=tape.result)
    y_orig = y
    y_fargs = [tape[y].fn; tape[y].args...]
    is_rrule_based = haskey(tape.c.pullbacks, y)
    if is_rrule_based
        # use rrule instead
        y = tape[y].args[1]
        y_fargs = is_kwfunc(y._op.fn) ? tape[y].args[3:end] : tape[y].args[2:end]
    end
    y_todo = [x for x in y_fargs if x isa V && tape[x] isa Call]
    x_todos = [todo_list(tape, x) for x in y_todo]
    # include y itself (original), its parents and their parents recursively
    todo = [[y_orig]; y_todo; vcat(x_todos...)]
    # deduplicate
    todo = collect(Set([bound(tape, v) for v in todo]))
    todo = sort(todo, by=v->v.id, rev=true)
    return todo
end


"""
Make a single step of backpropagation.
"""
function step_back!(tape::Tape, y::Variable)
    @debug "step_back!() for $(tape[y])"
    df = get_deriv_function(call_signature(tape, tape[y]))
    df isa NoTangent && return  # don't propagate deriative
    dy = tape.c.derivs[y]
    if df !== nothing
        # Yota rules
        @debug "Found Yota derivative: $df"
        y_fargs = [tape[y].fn; tape[y].args...]
        dxs = push!(tape, mkcall(df, dy, y_fargs...))
    elseif haskey(tape.c.pullbacks, y)
        # ChainRules
        pb = tape.c.pullbacks[y]
        @debug "Found pullback: $pb"
        dxs = push!(tape, mkcall(pb, dy))
        # propage derivs to rrule variable
        rr = tape[y].args[1]
        y_fargs = is_kwfunc(rr._op.fn) ? tape[rr].args[3:end] : tape[rr].args[2:end]
    else
        sig_str = join(["::$T" for T in Ghost.call_signature(tape, tape[y]).parameters], ", ")
        error("No deriative rule found for op $(tape[y]), " *
              "try defining it using \n\n\tChainRulesCore.rrule($sig_str) = ...\n")
    end
    for (i, x) in enumerate(y_fargs)
        if x isa V
            dx = push!(tape, mkcall(getfield, dxs, i))
            @debug "Updating derivative: $x -> $dx"
            set_or_add_deriv!(tape, x, dx)
            # if tape[x] isa Call
            #     push!(deriv_todo, x)
            # end
            # @debug "deriv_todo = $(join(deriv_todo, ", "))"
        end
    end
end


const BACKPROP_STATE = Ref{Tuple}()


"""
Backpropagate through the tape, record derivatives as new operations.
"""
function back!(tape::Tape; seed=1)
    # z - final variable (usually a loss)
    # y - resulting variable of current op
    # x - dependencies of y
    # dy - derivative of z w.r.t. y
    z = tape.result
    if (seed == 1) && (ndims(tape[z].val) > 0)
        error("Gradient of a vector-valued function requires a seed")
    end
    dy = push!(tape, Constant(seed))
    # save seed var to use in compilation later
    tape.meta[:seed] = dy
    # set initial derivative value
    tape.c.derivs[z] = dy
    # queue of variables to calculate derivatives for
    deriv_todo = todo_list(tape)
    for y in deriv_todo
        try
            step_back!(tape, y)
        catch e
            BACKPROP_STATE[] = (tape, y, deriv_todo)
            rethrow(e)
        end
    end
end


"""
    gradtape(f::Union{Function, DataType}, args...; seed=1)
    gradtape!(tape::Tape; seed=1)

Calculate and record to the tape gradients of `tape[tape.resultid]` w.r.t. `Input` nodes.
See grad() for more high-level API.
"""
function gradtape!(tape::Tape; seed=1)
    # update chainrules if number of rrule or no_rrule methods has changed
    update_chainrules_primitives!(force=false)
    # apply transformations needed for ChainRules
    chainrules_transform!(tape)
    # backpropagate gradients
    back!(tape; seed=seed)
    # add a tuple of (val, (gradients...))
    deriv_vars = [hasderiv(tape, v) ? getderiv(tape, v) : ZeroTangent() for v in inputs(tape)]
    deriv_tuple = push!(tape, mkcall(tuple, deriv_vars...))
    deriv_tuple_unthunked = push!(tape, mkcall(map, ChainRules.unthunk, deriv_tuple))
    new_result = push!(tape, mkcall(tuple, tape.result, deriv_tuple_unthunked))
    tape.result = new_result
    return tape
end


function gradtape(f::Union{Function,DataType}, args...; seed=1)
    _, tape = trace(f, args...; is_primitive=is_primitive, ctx=GradCtx())
    tape = gradtape!(tape; seed=seed)
    return tape
end


"Like Ghost.compile, but adds Yota specific ops"
function grad_compile(tape::Tape)
    ex = Ghost.to_expr(tape)
    seed_var = tape.meta[:seed]
    seed_default_val = tape[seed_var].val
    insert!(ex.args[1].args, 2, Expr(:parameters, Expr(:kw, :seed, seed_default_val)))
    seed_var_name = Ghost.make_name(seed_var.id)
    op_exprs = ex.args[2].args
    for op_ex in op_exprs
        if op_ex.args[1] == seed_var_name
            op_ex.args[2] = :seed
            break
        end
    end
    return Base.eval(@__MODULE__, ex)
end


const GRAD_CACHE = Dict{Any,Any}()

reset!() = empty!(GRAD_CACHE)


"""
    grad(f, args...; seed=1)

Find gradient of a callable `f` w.r.t. its arguments.

`grad()` returns two things: value of `f(args...)` and a tuple of
grafients w.r.t. to its inputs (including the callable itself).

```jldoctest
val, g = grad(x -> sum(x .+ 1), [1.0, 2.0, 3.0])

# output
(9.0, (ZeroTangent(), [1.0, 1.0, 1.0]))
```

By default, `grad()` expects the callable to return a scalar.
Vector-valued functions can be differentiated if a seed (starting value)
is provided. Seed is equivalent to the vector in VJP notation.

```jldoctest
val, g = grad(x -> 2x, [1.0, 2.0, 3.0]; seed=ones(3))

# output
([2.0, 4.0, 6.0], (ZeroTangent(), [2.0, 2.0, 2.0]))
```

All gradients can be applied to original variables using `update!()` function.

See also: [gradtape](@ref)
"""
function grad(f::Union{Function,DataType}, args...; seed=1)
    # key consists of function type and type of argument (for structs) or its size
    cache_key = (f, ([isstruct(arg) ? typeof(arg) : size(arg) for arg in args]...,))
    if haskey(GRAD_CACHE, cache_key)
        gf = GRAD_CACHE[cache_key]
        return gf(f, args...; seed=seed)
    else
        tape = gradtape(f, args...; seed=seed)
        gf = grad_compile(tape)
        GRAD_CACHE[cache_key] = gf
        return tape.retval
    end
end
