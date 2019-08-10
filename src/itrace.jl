import JuliaInterpreter
import JuliaInterpreter: enter_call, step_expr!, next_call!, @lookup, Frame, SSAValue, SlotNumber


getexpr(fr::Frame, pc::Int) = fr.framecode.src.code[pc]
current_expr(fr::Frame) = getexpr(fr, fr.pc)


"""
Split JuliaInterpreter call expression into a tuple of 3 elements:

 * function to be called
 * args to this function
 * vars on the tape corresponding to these args

If arguments include free parameters (not SlotNumber or SSAValue), these are recorded
to the tape as constants
"""
function split_int_call!(tape::Tape, fr::Frame, frame_vars::Dict, ex)
    arr = Meta.isexpr(ex, :(=)) ? ex.args[2].args : ex.args
    # for whatever reason JuliaInterpreter wraps some nodes in the original code into QuoteNode
    arr = [isa(x, QuoteNode) ? x.value : x for x in arr]
    cf = @lookup(fr, arr[1])
    cargs = [@lookup(fr, x) for x in arr[2:end]]
    cvars = Vector{Int}(undef, length(cargs))
    for (i, x) in enumerate(arr[2:end])
        # if isa(x, JuliaInterpreter.SlotNumber) || isa(x, JuliaInterpreter.SSAValue)
        if haskey(frame_vars, x)
            cvars[i] = frame_vars[x]
        else
            val = @lookup(fr, x)
            id = record!(tape, Constant, val)
            cvars[i] = id
            if val != x
                # if constant appeared to be a SlotNumber or SSAValue
                # store its mapping into frame_vars
                frame_vars[x] = id
            end
        end
    end
    return cf, cargs, cvars
end


"""
Given a Frame and current expression, extract LHS location (SlotNumber or SSAValue)
"""
get_location(fr::Frame, ex) = Meta.isexpr(ex, :(=)) ? ex.args[1] : JuliaInterpreter.SSAValue(fr.pc)

is_int_call_expr(ex) = Meta.isexpr(ex, :call) || (Meta.isexpr(ex, :(=)) && Meta.isexpr(ex.args[2], :call))


function itrace!(f, tape::Tape, argvars...; primitives)
    args, vars = zip(argvars...)
    fr = enter_call(f, args...)
    frame_vars = Dict{Any, Int}(JuliaInterpreter.SlotNumber(i + 1) => v for (i, v) in enumerate(vars))
    is_int_call_expr(current_expr(fr)) || next_call!(fr)  # skip non-call expressions
    ex = current_expr(fr)
    while !Meta.isexpr(ex, :return)
        # println("--------------- $ex -------------")
        if is_int_call_expr(ex)
            cf, cargs, cvars = split_int_call!(tape, fr, frame_vars, ex)
            loc = get_location(fr, ex)
            if cf isa UnionAll && cf <: NamedTuple
                # replace cf with namedtuple function, adjust arguments
                names = collect(cf.body.parameters)[1]
                cf = namedtuple
                cargs = [names; cargs]
                names_var_id = record!(tape, Constant, names)
                cvars = [names_var_id; cvars]
            end
            if cf in primitives || isa(cf, Core.Builtin) || isa(cf, Core.IntrinsicFunction)
                next_call!(fr)
                retval = @lookup(fr, loc)
                ret_id = record!(tape, Call, retval, cf, cvars)
                frame_vars[loc] = ret_id  # for slots it may overwrite old mapping
            else
                retval, ret_id = itrace!(cf, tape, zip(cargs, cvars)...; primitives=primitives)
                frame_vars[loc] = ret_id  # for slots it may overwrite old mapping
                next_call!(fr)  # can we avoid this double execution?
            end
        end
        ex = current_expr(fr)
    end
    retval = @lookup(fr, ex.args[1])
    ret_id = frame_vars[ex.args[1]]
    return retval, ret_id  # return var ID of a result variable
end


"""
Trace function f with arguments args using JuliaInterpreter
"""
function itrace(f, args...; primitives=PRIMITIVES, optimize=true)
    tape = Tape(guess_device(args))
    argvars = Vector(undef, length(args))
    for (i, arg) in enumerate(args)
        id = record!(tape, Input, arg)
        argvars[i] = (arg, id)
    end
    val, resultid = itrace!(f, tape, argvars...; primitives=primitives)
    tape.resultid = resultid
    if optimize
        tape = simplify(tape)
    end
    return val, tape
end

# TODO: remove Call.kwargs
