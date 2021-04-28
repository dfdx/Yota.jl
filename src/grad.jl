########################################################################
#                            GRAD CONTEXT                              #
########################################################################

# TODO: use it in tape instead of hardcoded fields
# the issue is with rebind_fields!() which must know context details
# perhaps introduce rebind_context!() wich is no-op by default?
struct GradContext
    # map from primal var to its pullback var
    # note: LittleDict is required because vars are mutable
    pullbacks::LittleDict{Variable, Variable}
    # map from primal var to its derivative var
    derivs::LittleDict{Variable, Variable}
end


########################################################################
#                              GRAD                                    #
########################################################################

const DEBUG_STATE = Ref{Any}()


getderiv(tape::Tape, v::Variable) = get(tape.derivs, bound(tape, v), nothing)
setderiv!(tape::Tape, x::Variable, dx::Variable) = (
    tape.derivs[bound(tape, x)] = bound(tape, dx)
)
hasderiv(tape::Tape, v::Variable) = getderiv(tape, v) !== nothing


function set_or_add_deriv!(tape::Tape, x::Variable, dx::Variable)
    if !hasderiv(tape, x)
        setderiv!(tape, x, dx)
    else
        old_dx = getderiv(tape, x)
        if tape[dx].val isa Composite || tape[old_dx].val isa Composite
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


# function step_back!(tape::Tape, op::Union{Call}, i::Int)
#     y = op
#     dy = getderiv(tape, y)
#     dy !== nothing || return           # op is not part of computation graph
#     op.args[i] isa Variable || return  # constant arg
#     x = tape[op.args[i].id]
#     if dy.val isa Zero
#         # propagate zero to dx (reuse dy node)
#         set_or_add_deriv!(tape, x, dy)
#         return
#     end
#     # TODO: finish
# end

"""
Transofrm the tape replacing calls `fn(args...)` for which `ChainRule.rrule` is defined
with the following chain or calls:

    * t = rrule(fn, args...)
    * val = getfield(t, 1)
    * pb = getfield(t, 2)

where `val = fn(args...)` and `pb` is the pullback function.
"""
function chainrules_transform!(tape::Tape)
    i = 1
    while i <= length(tape)
        op = tape[V(i)]
        if op isa Call && is_chainrules_primitive(call_signature(tape, op))
            rr_op = mkcall(rrule, op.fn, op.args...)
            val_op = mkcall(getfield, V(rr_op), 1)
            pb_op = mkcall(getfield, V(rr_op), 2)
            tape.pullbacks[V(val_op)] = V(pb_op)
            replace!(tape, i => [rr_op, val_op, pb_op]; rebind_to=2)
            i += 3  # rrule + 2 getfield ops
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
    @assert haskey(tape.pullbacks, y) "No pullback for op $(tape[y])"
    pb = tape.pullbacks[y]
    dxs = push!(tape, mkcall(pb, y))
    # propage derivs to rrule variable
    rr = tape[y].args[1]
    rr_fargs = tape[rr].args
    for (i, x) in enumerate(rr_fargs)
        if x isa V
            dx = push!(tape, mkcall(getfield, dxs, i))
            set_or_add_deriv!(tape, x, dx)
            if tape[x] isa Call
                push!(deriv_todo, x)
            end
        end
    end
end


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
    tape.derivs[z] = dy
    # queue of variables to calculate derivatives for
    deriv_todo = V[z]
    while !isempty(deriv_todo)
        y = popfirst!(deriv_todo)
        step_back!(tape, y, deriv_todo)
    end
end


"""
For each input that has a derivative on this tape check if the derivative
has the same size as the input.
"""
function check_deriv_sizes(tape::Tape)
    for (var_id, grad_var_id) in tape.derivs   # TODO: apply to pb_derivs as well
        # type of var and grad var may differ e.g. when grad_var is Zero()
        # if !isstruct(tape[var_id].val) && !isstruct(tape[grad_var_id].val)
        if tape[var_id].val isa AbstractArray && tape[grad_var_id].val isa AbstractArray
            var_size = size(tape[var_id].val)
            grad_var_size = size(tape[grad_var_id].val)
            if  var_size != grad_var_size
                @warn "Gradient %$grad_var_id has size $grad_var_size, " *
                    "but original variable %$var_id has size $var_size"
            end
        end
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
    # consistency check
    # check_deriv_sizes(tape)
    # add a tuple of (val, (gradients...))
    deriv_vars = [hasderiv(tape, v) ? getderiv(tape, v) : Zero() for v in inputs(tape)]
    deriv_tuple = push!(tape, mkcall(tuple, deriv_vars...))
    new_result = push!(tape, mkcall(tuple, tape.result, deriv_tuple))
    tape.result = new_result
    return tape
end


function gradtape(f::Union{Function, DataType}, args...)
    val, tape = trace(f, args...)
    tape = gradtape!(tape)
    return tape
end


const GRAD_CACHE = Dict{Any, Tape}()


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
function grad(f::Union{Function, DataType}, args...)
    # key consists of function type and type of argument (for structs) or its size
    cache_key = (f, ([isstruct(arg) ? typeof(arg) : size(arg) for arg in args]...,))
    if haskey(GRAD_CACHE, cache_key)
        # TODO: use cached function, don't cache the tape
        tape = GRAD_CACHE[cache_key]
        return play!(tape, args...)
    else
        # TODO: create a function, cache it, dismiss the tape
        tape = gradtape(f, args...)
        # compile!(tape)
        GRAD_CACHE[cache_key] = tape
        return tape[V(length(tape))].val
    end
end


"""
Non-caching version of grad(f, args...)
"""
function _grad(f::Union{Function, DataType}, args...)
    tape = gradtape(f, args...)
    return tape[V(length(tape))].val
end
