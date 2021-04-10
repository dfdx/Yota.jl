import IRTools
import IRTools: IR, @dynamo, self, insertafter!

# Some abbreviations used in this file
# * sid : SSA ID, ID of a variable in SSA form of IR
# * tid : Tape ID, ID of a variable on a Tape
# * fn : function being called
# * args : arguments to a function
# * fargs : array of [fn, args...]
# * farg_defs : SSA definitions of fargs, i.e. IRTools.Variable or objects
# * res / ret - result or return value


function __new__(T, args...)
    # note: we also add __new__() to the list of primitives so it's not overdubbed recursively
    if T <: NamedTuple
        return T(args)
    else
        return T(args...)
    end
end


# __tuple__(args...) = tuple(args...)
# __getfield__(args...) = getfield(args...)


function module_functions(modl)
    res = Vector{Function}()
    for s in Base.names(modl; all=true)
        isdefined(modl, s) || continue
        fn = getfield(modl, s)
        if fn isa Function # && match(r"^[a-z#]+$", string(s)) != nothing
            push!(res, fn)
        end
    end
    return res
end

const BASE_PRIMITIVE_FUNCTIONS = Set{Any}(vcat(
    module_functions(Base),
    module_functions(Core),
    module_functions(Core.Intrinsics),
    [Broadcast.materialize, Broadcast.broadcasted, Colon(), (:),
     Base.not_int,
     # our own special functions
     __new__, namedtuple, guess_device]));


const PRIMITIVES = BASE_PRIMITIVE_FUNCTIONS


################################################################################
#                                Frame                                         #
################################################################################

"""Frame of a call stack"""
mutable struct Frame
    # IR var ID to Tape var ID. Note that IR ID refers to the _original_ IR,
    # not the transformed one. In fact, any unique name of SSA instructions
    # would fit, using IR IDs is just convenient
    ir2tape::Dict{Int, Int}
    # result ID - tape ID corresponding to latest return value
    # from this call frame
    resultid::Int
end

function Base.show(io::IO, fr::Frame)
    map_str = join(["$k=>$v" for (k, v) in fr.ir2tape], ",")
    print(io, "Frame($map_str, $(fr.resultid))")
end


################################################################################
#                        IRTracer (defs and utils)                             #
################################################################################

mutable struct IRTracer
    primitives::Set{Any}
    tape::Tape
    frames::Vector{Frame}
end

function IRTracer(;primitives=PRIMITIVES)
    tape = Tape()
    return IRTracer(primitives, tape, [])
end

Base.show(io::IO, t::IRTracer) = print(io, "IRTracer($(length(t.tape)))")


promote_const_value(x::QuoteNode) = x.value
promote_const_value(x::GlobalRef) = getproperty(x.mod, x.name)
promote_const_value(x) = x


function get_tape_vars(t::IRTracer, arg_defs::Union{Vector, Tuple})
    result = Vector{Any}(undef, length(arg_defs))
    for (i, arg) in enumerate(arg_defs)
        if arg isa IRTools.Variable
            tape_id = t.frames[end].ir2tape[arg.id]
            result[i] = Variable(tape_id)   # Yota.Variable
        else
            val = promote_const_value(arg)
            # arg_var = record!(t.tape, Constant, val)
            result[i] = val
        end
    end
    return result
end


"""Push a new call frame to tracer, setting function params accordingly"""
function push_frame!(t::IRTracer, arg_defs...)
    tape_vars = get_tape_vars(t, arg_defs)
    frame = Frame(
        Dict(i => v.id for (i, v) in enumerate(tape_vars) if v isa Variable),
        -1
    )
    push!(t.frames, frame)
end


"""Pop call frame from tracer"""
function pop_frame!(t::IRTracer, res_sid::Int)
    frame = pop!(t.frames)
    # create mapping from the current SSA ID to the last instruction on the tape
    t.frames[end].ir2tape[res_sid] =
        (frame.resultid == -1 ? length(t.tape) : frame.resultid)
end


"""Set target branch parameters to variables corresponding to SSA args"""
function set_branch_params!(t::IRTracer, ssa_args, target_params)
    tape_vars = get_tape_vars(t, ssa_args)
    ir2tape = t.frames[end].ir2tape
    for (v, p) in zip(tape_vars, target_params)
        ir2tape[p] = v.id
    end
end


"""Set return variable for the current frame"""
function set_return!(t::IRTracer, arg_sid_ref)
    # global STATE = (t, arg_sid_ref)
    tape_var = get_tape_vars(t, [arg_sid_ref[]])[1]
    t.frames[end].resultid = tape_var.id
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
* farg_defs - IR variables of the operation
* fargs - values of the operation
"""
function record_or_recurse!(t::IRTracer, res_sid::Int, farg_defs, fargs...)
    fn, args = fargs[1], fargs[2:end]
    # global STATE = (t, res_sid, farg_defs, fargs)
    if fn in t.primitives || (fn isa Type && fn <: NamedTuple)
        res = fn(args...)
        tape_vars = get_tape_vars(t, farg_defs[2:end])
        # record corresponding op to the tape
        res_tid = record!(t.tape, Call, res, fn, tape_vars)
        # update mapping from SSA var to tape var
        # note that in functions with loops this mapping may change over time
        t.frames[end].ir2tape[res_sid] = res_tid
    else
        push_frame!(t, farg_defs...)
        res = t(fn, args...)
        pop_frame!(t, res_sid)
    end
    return res
end


function record_const!(t::IRTracer, res_sid, val)
    val = val isa QuoteNode ? val.value : val
    res_tid = record!(t.tape, Constant, val)
    t.frames[end].ir2tape[res_sid] = res_tid
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
Trace function call, produce call value and a Tape
"""
function trace(f, args...; primitives=PRIMITIVES)
    t = IRTracer(; primitives=primitives)
    for arg in args
        record!(t.tape, Input, arg)
    end
    push!(t.frames, Frame(Dict(i + 1 => i for i in 1:length(args)), -1))
    val = t(f, args...)
    t.tape.resultid = t.frames[1].resultid
    tape = t.tape
    # if optimize
    #     tape = simplify(tape)
    # end
    return val, tape
end