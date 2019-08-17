import JuliaInterpreter
import JuliaInterpreter: enter_call, step_expr!, @lookup, Frame, SSAValue, SlotNumber


getexpr(frame::Frame, pc::Int) = frame.framecode.src.code[pc]
current_expr(frame::Frame) = getexpr(frame, frame.pc)


"""
Split JuliaInterpreter call expression into a tuple of 2 elements:

 * callable (e.g. function or callable struct) & its arguments
 * vars on the tape corresponding to these function & args

If arguments include free parameters (not SlotNumber or SSAValue), these are recorded
to the tape as constants
"""
function split_int_call!(tape::Tape, frame::Frame, frame_vars::Dict, ex)
    arr = Meta.isexpr(ex, :(=)) ? ex.args[2].args : ex.args
    # for whatever reason JuliaInterpreter wraps some nodes in the original code into QuoteNode
    arr = [isa(x, QuoteNode) ? x.value : x for x in arr]
    cfargs = [x isa Symbol ? x : @lookup(frame, x) for x in arr]
    cvars = Vector{Int}(undef, length(cfargs))
    for (i, x) in enumerate(arr)
        # if isa(x, JuliaInterpreter.SlotNumber) || isa(x, JuliaInterpreter.SSAValue)
        if haskey(frame_vars, x)
            cvars[i] = frame_vars[x]
        else
            val = x isa Symbol ? x : @lookup(frame, x)
            id = record!(tape, Constant, val)
            cvars[i] = id
            if val != x
                # if constant appeared to be a SlotNumber or SSAValue
                # store its mapping into frame_vars
                frame_vars[x] = id
            end
        end
    end
    return cfargs, cvars
end


"""
Given a Frame and current expression, extract LHS location (SlotNumber or SSAValue)
"""
get_location(frame::Frame, ex) = Meta.isexpr(ex, :(=)) ? ex.args[1] : JuliaInterpreter.SSAValue(frame.pc)

is_int_call_expr(ex) = Meta.isexpr(ex, :call) || (Meta.isexpr(ex, :(=)) && Meta.isexpr(ex.args[2], :call))
is_int_assign_expr(ex) = Meta.isexpr(ex, :(=)) && (isa(ex.args[2], SlotNumber) || isa(ex.args[2], SSAValue))

is_interesting_expr(ex) = is_int_call_expr(ex) || is_int_assign_expr(ex) || Meta.isexpr(ex, :return)

# f, f_var, args, vars
# f_arg_vars?
function itrace!(tape::Tape, fargs, vars; primitives)
    frame = enter_call(fargs...)
    frame_vars = Dict{Any, Int}(JuliaInterpreter.SlotNumber(i) => v for (i, v) in enumerate(vars))
    # f might be a callable struct, so we need to record it and add to frame_vars
    # f_id = record!(tape, Constant, f)  # should not be constant! should caller record it?
    # frame_vars[SlotNumber(1)] = f
    is_interesting_expr(current_expr(frame)) || step_expr!(frame)  # skip non-call expressions
    ex = current_expr(frame)
    while !Meta.isexpr(ex, :return)
        if is_int_assign_expr(ex)
            lhs, rhs = ex.args
            frame_vars[lhs] = frame_vars[rhs]
            step_expr!(frame)
        elseif is_int_call_expr(ex)
            # read as "current function & arguments", "current variables"
            cfargs, cvars = split_int_call!(tape, frame, frame_vars, ex)
            cf = cfargs[1]
            loc = get_location(frame, ex)
            # there are several special cases such as NamedTuples and constructors
            # we replace these with calls to special helper functions
            if cf isa UnionAll && cf <: NamedTuple
                # replace cf with namedtuple function, adjust arguments
                names = collect(cf.body.parameters)[1]
                cf = namedtuple
                insert!(cfargs, 2, names)
                names_var_id = record!(tape, Constant, names)
                cvars = [names_var_id; cvars]
            elseif cf isa DataType
                # constructor, replace with a call to __new__ which we know how to differentiate
                T = cf
                cf = __new__
                insert!(cfargs, 2, T)
                T_var_id = record!(tape, Constant, T)
                cvars = [T_var_id; cvars]
            elseif cf == Base.tuple
                cf = __tuple__
            elseif cf == Base.getfield || (cf == Base.getindex && isa(tape[cvars[2]].val, NamedTuple))
                # similar to constuctors, there's a special case for __getfield__ in backprop
                cf = __getfield__
            end
            # if current function is a primitive of a built-in, write it to the tape
            # otherwise recurse into the current function
            if cf in primitives || isa(cf, Core.Builtin) || isa(cf, Core.IntrinsicFunction)
                step_expr!(frame)
                retval = @lookup(frame, loc)
                ret_id = record!(tape, Call, retval, cf, cvars[2:end])
                frame_vars[loc] = ret_id  # for slots it may overwrite old mapping
            else
                try
                    retval, ret_id = itrace!(tape, cfargs, cvars; primitives=primitives)
               catch
                    println("Failed to trace through function $cf")
                    rethrow()
                end
                frame_vars[loc] = ret_id  # for slots it may overwrite old mapping
                step_expr!(frame)  # can we avoid this double execution?
            end
        else
            step_expr!(frame)
        end
        ex = current_expr(frame)
    end
    retval = @lookup(frame, ex.args[1])
    ret_id = frame_vars[ex.args[1]]
    return retval, ret_id  # return var ID of a result variable
end


"""
Trace function f with arguments args using JuliaInterpreter
"""
function itrace(f, args...; primitives=PRIMITIVES, optimize=true)
    tape = Tape(guess_device(args))
    # record arguments as input variables
    fargs = Vector(undef, length(args) + 1)
    vars = Vector{Int}(undef, length(args) + 1)
    for (i, arg) in enumerate([f, args...])
        id = record!(tape, Input, arg)
        fargs[i] = arg
        vars[i] = i
    end
    val, resultid = itrace!(tape, fargs, vars; primitives=primitives)
    tape.resultid = resultid
    if optimize
        tape = simplify(tape)
    end
    return val, tape
end
