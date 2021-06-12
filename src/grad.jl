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
                    mkcall(rrule, op.fn, op.args...))
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
function step_back!(tape::Tape, y::Variable, deriv_todo::Vector{Variable})
    # TODO: here's the problem with the constructor_loss test:
    # we reach y = __new__(...) twice and update all its fields twice
    @debug "step_back!() for $(tape[y])"
    df = get_deriv_function(call_signature(tape, tape[y]))
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
        y_fargs = is_kwfunc(rr._op.fn) ? tape[rr].args[3:end] : tape[rr].args
    else
        error("Neither ChainRules pullback, nor native Yota " *
              "derivative found for op $(tape[y])")
    end
    for (i, x) in enumerate(y_fargs)
        if x isa V
            dx = push!(tape, mkcall(getfield, dxs, i))
            @debug "Updating derivative: $x -> $dx"
            set_or_add_deriv!(tape, x, dx)
            if tape[x] isa Call
                push!(deriv_todo, x)
            end
            @debug "deriv_todo = $(join(deriv_todo, ", "))"
        end
    end
end


const BACKPROP_STATE = Ref{Tuple}()


"""
Backpropagate through the tape, record derivatives as new operations.
"""
function back!(tape::Tape)
    # z - final variable (usually a loss)
    # y - resulting variable of current op
    # x - dependencies of y
    # dy - derivative of z w.r.t. y
    z = tape.result
    # using one() of type of the result for seed to keep type stability
    @assert ndims(tape[z].val) == 0 "Function must return scalar!"
    dy = push!(tape, Constant(one(tape[z].val)))
    # set initial derivative value
    tape.c.derivs[z] = dy
    # queue of variables to calculate derivatives for
    deriv_todo = V[z]
    deriv_processed = Set{V}()
    while !isempty(deriv_todo)
        y = popfirst!(deriv_todo)
        ## skip double calculation for the same var in multupath graphs
        if y in deriv_processed
            @debug "Already processed $y, skipping it"
            continue
        end
        try
            step_back!(tape, y, deriv_todo)
        catch e
            BACKPROP_STATE[] = (tape, y, deriv_todo)
            rethrow(e)
        end
        push!(deriv_processed, y)
    end
end


"""
    gradtape(f::Union{Function, DataType}, args...)
    gradtape!(tape::Tape)

Calculate and record to the tape gradients of `tape[tape.resultid]` w.r.t. `Input` nodes.
See grad() for more high-level API.
"""
function gradtape!(tape::Tape)
    # apply transformations needed for ChainRules
    chainrules_transform!(tape)
    # backpropagate gradients
    back!(tape)
    # add a tuple of (val, (gradients...))
    deriv_vars = [hasderiv(tape, v) ? getderiv(tape, v) : ZeroTangent() for v in inputs(tape)]
    deriv_tuple = push!(tape, mkcall(tuple, deriv_vars...))
    deriv_tuple_unthunked = push!(tape, mkcall(map, ChainRules.unthunk, deriv_tuple))
    new_result = push!(tape, mkcall(tuple, tape.result, deriv_tuple_unthunked))
    tape.result = new_result
    return tape
end


function gradtape(f::Union{Function,DataType}, args...)
    _, tape = trace(f, args...; is_primitive=is_primitive, ctx=GradCtx())
    tape = gradtape!(tape)
    return tape
end


const GRAD_CACHE = Dict{Any,Any}()


"""
    grad(f, args...)

Find gradient of `f` w.r.t. its arguments.
Example:

    val, g = grad(sum, rand(3))

where:
  - val is the value of `f` at this point
  - g is a tuple of gradients

All gradients can be applied to original variables using `update!()` function.

See also: gradtape
"""
function grad(f::Union{Function,DataType}, args...)
    # key consists of function type and type of argument (for structs) or its size
    cache_key = (f, ([isstruct(arg) ? typeof(arg) : size(arg) for arg in args]...,))
    if haskey(GRAD_CACHE, cache_key)
        gf = GRAD_CACHE[cache_key]
        return gf(f, args...)
    else
        tape = gradtape(f, args...)
        gf = compile(tape)
        GRAD_CACHE[cache_key] = gf
        return tape.retval
    end
end
