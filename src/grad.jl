###############################################################################
#                                  GRAD CONTEXT                               #
###############################################################################

struct ChainRulesCtx end


function has_rrule(f, args...)
    F = Core.Typeof(f)
    Args = Core.Typeof.(args)
    Core.Compiler.return_type(rrule, Tuple{YotaRuleConfig, F, Args...}) !== Nothing && return true
    if is_kwfunc(F)
        Args_kwrrule = Tuple{Any, typeof(Core.kwfunc(f)), YotaRuleConfig, Args...,}
        Core.Compiler.return_type(Core.kwfunc(rrule), Args_kwrrule) !== Nothing && return true
    end
    return false
end

Umlaut.isprimitive(::ChainRulesCtx, f, args...) = has_rrule(f, args...)


struct GradCtx
    # map from primal var to its pullback var
    pullbacks::Dict{Variable,Variable}
    # map from primal var to its derivative var
    derivs::Dict{Variable,Variable}
end

GradCtx() = GradCtx(Dict(), Dict())

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


const BASE_CTX = BaseCtx()
const YOTA_RULE_CONFIG = YotaRuleConfig()

"""
    actually_tracable(f, args...)

Add exception for `isprimitive(BaseCtx(), f, args...)`.
Note that it does NOT cancel `isprimitive(ChainRulesCtx(), f, args...)`.
"""
actually_tracable(f, args...) = false
# we want to trace broadcast(ed) down to their version with rrules.
actually_tracable(::typeof(broadcast), args...) = true
actually_tracable(::typeof(broadcasted), args...) = true


function Umlaut.isprimitive(::GradCtx, f, args...)
    if isprimitive(BaseCtx(), f, args...) && !actually_tracable(f, args...)
        return true
    end
    if isprimitive(ChainRulesCtx(), f, args...)
        return true
    end
    return false
end


###############################################################################
#                            BCAST GRAD CONTEXT                               #
###############################################################################

struct BcastGradCtx
    inner
end


# get_static_params is broken for BcastGradCtx, so turning off
# this feature for now
Umlaut.get_static_params(::Tracer{BcastGradCtx}, v_fargs::Union{Tuple, Vector}) = Core.svec([])

function Umlaut.code_signature(::BcastGradCtx, v_fargs)
    f, args... = Umlaut.var_values(v_fargs)
    return (f, map(eltype, args))
end


function Umlaut.trace_call!(t::Tracer{BcastGradCtx}, v_fargs...)
    fargs = Umlaut.map_vars(v -> v.op.val, v_fargs)
    f, args... = fargs
    # TODO: should we check isprimitive() for map(first, fargs) instead?
    el_args = map(first, fargs[2:end])
    if isprimitive(t.tape.c.inner, f, el_args...) && !Umlaut.is_ho_tracable(t.tape.c.inner, f, el_args...)
        rr_op = (is_kwfunc(f) ?
                    mkcall(Core.kwfunc(bcast_rrule), v_args[1], bcast_rrule, YOTA_RULE_CONFIG, broadcasted, v_args[2:end]...) :
                    mkcall(bcast_rrule, YOTA_RULE_CONFIG, broadcasted, v_fargs...))
        v_rr = push!(t.tape, rr_op)
        v_val = push!(t.tape, mkcall(_getfield, v_rr, 1))
        v_pb = push!(t.tape, mkcall(_getfield, v_rr, 2))
        t.tape.c.inner.pullbacks[v_val] = v_pb
        return v_val
    else
        return trace!(t, v_fargs)
    end
end


###############################################################################
#                                   GRAD                                      #
###############################################################################

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
            new_dx = push!(tape, mkcall(+, dx, old_dx))
        else
            new_dx = push!(tape, mkcall(broadcast, +, dx, old_dx))
        end
        setderiv!(tape, x, new_dx)
    end
end


"""
Collect variables that we need to step through during the reverse pass.
The returned vector is already deduplicated and reverse-sorted
"""
function todo_list(tape::Tape{GradCtx}, y=tape.result)
    y_orig = y
    @assert(tape[y] isa Call, "The tape's result is expected to be a Call, " *
            "but instead $(typeof(tape[tape.result])) was encountered")
    y_fargs = [tape[y].fn; tape[y].args...]
    is_rrule_based = haskey(tape.c.pullbacks, y)
    if is_rrule_based
        # use rrule instead
        y = tape[y].args[1]
        y_fargs = is_kwfunc(y._op.fn) ? tape[y].args[3:end] : tape[y].args
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

call_values(op::Call) = Umlaut.var_values([op.fn, op.args...])

"""
Transofrm the tape replacing calls `fn(args...)` for which `ChainRule.rrule` is defined
with the following chain or calls:
    * t = rrule(fn, args...)
    * val = _getfield(t, 1)
    * pb = _getfield(t, 2)
where `val = fn(args...)` and `pb` is the pullback function.
"""
function chainrules_transform!(tape::Tape)
    i = 1
    while i <= length(tape)
        op = tape[V(i)]
        if op isa Call && isprimitive(ChainRulesCtx(), call_values(op)...)
            # replace f(args...) with rrule(f, args...)
            v_f, v_args, line = op.fn, op.args, op.line
            f = op.fn isa V ? tape[op.fn].val : op.fn
            rr_op = if is_kwfunc(f)
                v_kw, v_orig_f, v_orig_args... = v_args
                mkcall(Core.kwfunc(rrule), v_kw, rrule, YOTA_RULE_CONFIG, v_orig_f, v_orig_args...; line=line)
            else
                mkcall(rrule, YOTA_RULE_CONFIG, v_f, v_args...; line=line)
            end
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
Make a single step of backpropagation.
"""
function step_back!(tape::Tape, y::Variable)
    @debug "step_back!() for $(tape[y])"
    if !hasderiv(tape, y)
        @debug "No derivative found for y = $y, stop propagating the gradient in this path"
        return
    end
    dy = tape.c.derivs[y]
    if haskey(tape.c.pullbacks, y)
        # ChainRules
        pb = tape.c.pullbacks[y]
        @debug "Found pullback: $pb"
        dxs = push!(tape, mkcall(pb, dy))
        # propage derivs to rrule variable
        rr = tape[y].args[1]
        y_fargs = is_kwfunc(rr._op.fn) ? tape[rr].args[4:end] : tape[rr].args[2:end]
    else
        sig_str = join(["::$T" for T in Umlaut.call_signature(tape, tape[y]).parameters], ", ")
        error("No deriative rule found for op $(tape[y]), " *
              "try defining it using \n\n\tChainRulesCore.rrule($sig_str) = ...\n")
    end
    for (i, x) in enumerate(y_fargs)
        if x isa V
            dx = push!(tape, mkcall(getfield, dxs, i))
            @debug "Updating derivative: $x -> $dx"
            set_or_add_deriv!(tape, x, dx)
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
    elseif seed == :auto
        zval = tape[z].val
        # @assert zval isa Number || zval isa AbstractArray
        seed = zval isa AbstractArray ? ones(eltype(zval), size(zval)) : one(zval)
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


function gradtape!(tape::Tape; seed=1)
    # apply transformations needed for ChainRules
    chainrules_transform!(tape)
    # backpropagate gradients
    back!(tape; seed=seed)
    # add a tuple of (val, (gradients...))
    deriv_vars = [hasderiv(tape, v) ? getderiv(tape, v) : ZeroTangent() for v in inputs(tape)]
    deriv_tuple = push!(tape, mkcall(tuple, deriv_vars...))
    # unthunk results
    deriv_tuple_unthunked = push!(tape, mkcall(map, ChainRules.unthunk, deriv_tuple))
    new_result = push!(tape, mkcall(tuple, tape.result, deriv_tuple_unthunked))
    # set result
    tape.result = new_result
    return tape
end


"""
    gradtape(f, args...; ctx=GradCtx(), seed=1)
    gradtape!(tape::Tape; seed=1)

Calculate and record to the tape gradients of `tape[tape.resultid]` w.r.t. `Input` nodes.
See grad() for more high-level API.
"""
function gradtape(f, args...; ctx=GradCtx(), seed=1)
    _, tape = trace(f, args...; ctx=ctx)
    # TODO: if it works, move the hack to Umlaut
    if tape[tape.result] isa Input
        new_result = push!(tape, mkcall(identity, tape.result))
        tape.result = new_result
    end
    tape = gradtape!(tape; seed=seed)
    return tape
end


"Like Umlaut.compile, but adds Yota specific ops"
function grad_compile(tape::Tape)
    ex = Umlaut.to_expr(tape)
    seed_var = tape.meta[:seed]
    seed_default_val = tape[seed_var].val
    insert!(ex.args[1].args, 2, Expr(:parameters, Expr(:kw, :seed, seed_default_val)))
    seed_var_name = Umlaut.make_name(seed_var.id)
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
using Yota   # hide

val, g = grad(x -> sum(x .+ 1), [1.0, 2.0, 3.0])

# output
(9.0, (ChainRulesCore.ZeroTangent(), [1.0, 1.0, 1.0]))
```

By default, `grad()` expects the callable to return a scalar.
Vector-valued functions can be differentiated if a seed (starting value)
is provided. Seed is equivalent to the vector in VJP notation.

```jldoctest
using Yota   # hide

val, g = grad(x -> 2x, [1.0, 2.0, 3.0]; seed=ones(3))

# output
([2.0, 4.0, 6.0], (ChainRulesCore.ZeroTangent(), [2.0, 2.0, 2.0]))
```

All gradients can be applied to original variables using `update!()` function.

See also: [gradtape](@ref)
"""
function grad(f, args...; seed=1)
    # key consists of function type and type of argument (for structs) or its size
    # cache_key = (f, ([(isstruct(arg) || args isa Function) ? typeof(arg) : size(arg) for arg in args]...,))
    cache_key = map(typeof, (f, args...))
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
