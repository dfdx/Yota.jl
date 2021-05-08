import IRTools
import IRTools: IR, @dynamo, self, insertafter!


function module_functions(modl)
    res = Vector{Function}()
    for s in Base.names(modl; all=true)
        isdefined(modl, s) || continue
        fn = getfield(modl, s)
        # && match(r"^[a-z#]+$", string(s)) != nothing
        if fn isa Function && !in(string(fn)[1], "#_@")
            push!(res, fn)
        end
    end
    return res
end

# include("primitives.jl")

const BASE_PRIMITIVE_FUNCTIONS = vcat(
    module_functions(Base),
    module_functions(Core),
    module_functions(Core.Intrinsics),
    [Broadcast.materialize, Broadcast.broadcasted, Colon(), (:),
     Base.not_int,
     # our own special functions
     __new__, namedtuple, guess_device]);


const PRIMITIVES = FunctionResolver{Bool}(
    collect(Tuple{typeof(f), Vararg} => true for f in BASE_PRIMITIVE_FUNCTIONS)
)


function is_primitive(sig)
    return (sig in PRIMITIVES ||
            is_yota_primitive(sig) ||
            is_chainrules_primitive(sig))
end


################################################################################
#                                Frame                                         #
################################################################################

"""Frame of a call stack"""
mutable struct Frame
    # IR var ID to Tape var. Note that IR ID refers to the _original_ IR,
    # not the transformed one. In fact, any unique name of SSA instructions
    # would fit, using IR IDs is just convenient.
    # Note that a constant value can be passed instead of a Tape var
    ir2tape::Dict{Int, Any}
    # result ID - tape ID corresponding to latest return value
    # from this call frame
    result::V
    # function and arguments - used purely for debugging
    fargs::Tuple
end

function Base.show(io::IO, fr::Frame)
    map_str = join(["$k=>$v" for (k, v) in fr.ir2tape], ",")
    print(io, "Frame($map_str, $(fr.result))")
end


################################################################################
#                        IRTracer (defs and utils)                             #
################################################################################

mutable struct IRTracer
    is_primitive::Function
    tape::Tape
    frames::Vector{Frame}
end

function IRTracer(;is_primitive=is_chainrules_primitive)
    tape = Tape()
    return IRTracer(is_primitive, tape, [])
end

Base.show(io::IO, t::IRTracer) = print(io, "IRTracer($(length(t.tape)))")


promote_const_value(x::QuoteNode) = x.value
promote_const_value(x::GlobalRef) = getproperty(x.mod, x.name)
promote_const_value(x) = x


function get_tape_vars(t::IRTracer, farg_irvars::Union{Vector, Tuple})
    result = Vector{Any}(undef, length(farg_irvars))
    for (i, arg) in enumerate(farg_irvars)
        if arg isa IRTools.Variable
            # cat be either tape var or constant value if a function
            # wall called with a constant
            x = t.frames[end].ir2tape[arg.id]
            result[i] = x isa V ? x : promote_const_value(x)
            # result[i] = V(t.tape[x])   # Yota.Variable
        else
            result[i] = promote_const_value(arg)
        end
    end
    return result
end


"""Push a new call frame to tracer, setting function params accordingly"""
function push_frame!(t::IRTracer, fargs, farg_irvars...)
    tape_vars = get_tape_vars(t, farg_irvars)
    frame = Frame(
        # TODO: if a constant is passed to the function,
        # tape doesn't contain a corresponding variable
        # (in previous versions there was a Constant node for this)
        # possible solution:
        Dict(i => v for (i, v) in enumerate(tape_vars)),
        # also make ir2tape accept values
        # Dict(i => v for (i, v) in enumerate(tape_vars) if v isa V),
        V(0),
        fargs,
    )
    push!(t.frames, frame)
end


"""Pop call frame from tracer"""
function pop_frame!(t::IRTracer, res_sid::Int)
    frame = pop!(t.frames)
    # create mapping from the current SSA ID to the last instruction on the tape
    t.frames[end].ir2tape[res_sid] =
        (frame.result.id == 0 ? bound(t.tape, V(length(t.tape))) : frame.result)
end


"""Set target branch parameters to variables corresponding to SSA args"""
function set_branch_params!(t::IRTracer, ssa_args, target_params)
    tape_vars = get_tape_vars(t, ssa_args)
    ir2tape = t.frames[end].ir2tape
    for (v, p) in zip(tape_vars, target_params)
        ir2tape[p] = v
    end
end


"""Set return variable for the current frame"""
function set_return!(t::IRTracer, arg_sid_ref)
    tape_var = get_tape_vars(t, [arg_sid_ref[]])[1]
    t.frames[end].result = tape_var
end


################################################################################
#                        IRTracer (body) + irtrace()                           #
################################################################################

function rewrite_special_cases!(ir::IR)
    for (v, st) in ir
        if Meta.isexpr(st.expr, :new)
            ir[v] = Expr(:call, __new__, st.expr.args...)
        end
    end
end


"""
Record function call onto a tape or recurse into it.

Params:
-------
* t::IRTracer - current tracer
* res_id::Int - IR ID of the operation
* farg_irvars - IR variables of the operation
* fargs - values of the operation
"""
function record_or_recurse!(t::IRTracer, res_sid::Int, farg_irvars, fargs...)
    fn, args = fargs[1], fargs[2:end]
    # global STATE = (t, res_sid, farg_irvars, fargs)
    # fn == Core.kwfunc(sum) && error()
    if t.is_primitive(Tuple{map(typeof, fargs)...})
        tape_vars = get_tape_vars(t, farg_irvars)
        # record corresponding op to the tape
        res = push!(t.tape, mkcall(tape_vars...))
        # update mapping from SSA var to tape var
        # note that in functions with loops this mapping may change over time
        t.frames[end].ir2tape[res_sid] = res
        val = t.tape[res].val
    else
        push_frame!(t, fargs, farg_irvars...)
        val = t(fn, args...)
        pop_frame!(t, res_sid)
    end
    return val
end


function record_const!(t::IRTracer, res_sid, val)
    val = val isa QuoteNode ? val.value : val
    res = push!(t.tape, Constant(val))
    t.frames[end].ir2tape[res_sid] = res
    return val
end


function trace_branches!(ir::IR)
    # if a block ends with a branch, we map its parameters to tape IDs
    # which currently correspond to argument SSA IDs
    for block in IRTools.blocks(ir)
        for branch in IRTools.branches(block)
            if IRTools.isreturn(branch)
                ret_v = branch.args[1]
                push!(ir, Expr(:call, set_return!, self, Ref(ret_v)))
            else
                ssa_args = branch.args
                target_params = [v.id for v in ir.blocks[branch.block].args]
                push!(block, Expr(:call, set_branch_params!, self, ssa_args, target_params))
            end
        end
    end
end


@dynamo function (t::IRTracer)(fargs...)
    ir = IR(fargs...)
    ir === nothing && return   # intrinsic functions
    # TODO (for loops): IRTools.expand!(ir)
    rewrite_special_cases!(ir)
    for (v, st) in ir
        ex = st.expr
        # note the difference:
        # * `ex.args` is an array and thus will be passed to a function as is,
        # including definitions of IRTools.Variable;
        # * `ex.args...` is top-level to this expression and thus all Variable's
        # will be replaced with actual values during runtime
        if Meta.isexpr(ex, :call)
            ir[v] = IRTools.xcall(record_or_recurse!, self, v.id, ex.args, ex.args...)
        else
            # e.g. GlobalRef
            # note: using insertafter!() due to
            # https://github.com/FluxML/IRTools.jl/issues/78
            # ir[v] = Expr(:call, record_const!, self, v.id, v)
            insertafter!(ir, v, IRTools.xcall(record_const!, self, v.id, v))
        end
    end
    trace_branches!(ir)
    return ir
end


"""
    trace(f, args...; is_primitive, primitives)

Trace function call, produce call value and a Tape.

`trace` records to the tape primitive methods and recursively dives into
non-primitives. There are 2 ways to tell `trace` that a particular method
is a primitive:

* provide `is_primitive(sig) -> Bool` function, where `sig` is
    is a method signature, e.g. `map(typeof, (f, args...))`
* provide an iterable `primitives`; in this case `trace` matches
    all methods of this function
"""
function trace(f, args...; is_primitive=is_primitive, primitives=nothing)
    if primitives !== nothing
        sigs = FunctionResolver{Bool}([Tuple{typeof(f), Vararg} => true for f in primitives])
        is_primitive = sig -> sig in sigs
    end
    t = IRTracer(; is_primitive=is_primitive)
    arg_vars = inputs!(t.tape, f, args...)
    frame = Frame(Dict(i => a for (i, a) in enumerate(arg_vars)), V(0), (f, args...))
    push!(t.frames, frame)
    val = t(f, args...)
    t.tape.result = t.frames[1].result
    tape = t.tape
    # if optimize
    #     tape = simplify(tape)
    # end
    return val, tape
end