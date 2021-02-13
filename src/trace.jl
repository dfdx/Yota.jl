function __new__(T, args...)
    # note: we also add __new__() to the list of primitives so it's not overdubbed recursively
    if T <: NamedTuple
        return T(args)
    else
        return T(args...)
    end
end


__tuple__(args...) = tuple(args...)
__getfield__(args...) = getfield(args...)


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

const PRIMITIVES = Set{Any}(vcat(
    module_functions(Base),
    module_functions(Core),
    module_functions(Core.Intrinsics),
    [Broadcast.materialize, Broadcast.broadcasted, Colon(), (:),
     Base.not_int,
     # our own special functions
     __new__, __tuple__, __getfield__, namedtuple, guess_device]));


################################################################################
################################################################################
#                           IRTools-based Tracer                               #
################################################################################
################################################################################

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


################################################################################
#                                Frame                                         #
################################################################################

"""Frame of a call stack"""
mutable struct Frame
    # SSA var ID to Tape var ID. Note that SSA ID refers to the _original_ IR,
    # not the transformed one. In fact, any unique name of SSA instructions
    # would fit, using SSA IDs is just convenient
    ssa2tape::Dict{Int, Int}
    # result ID - tape ID corresponding to latest return value
    # from this call frame
    resultid::Int
end

function Base.show(io::IO, fr::Frame)
    map_str = join(["$k=>$v" for (k, v) in fr.ssa2tape], ",")
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


function ssa_args_to_tape_vars!(t::IRTracer, arg_defs::Union{Vector, Tuple})
    result = Vector{Int}(undef, length(arg_defs))
    for (i, arg) in enumerate(arg_defs)
        if arg isa IRTools.Variable
            result[i] = t.frames[end].ssa2tape[arg.id]
        else
            val = promote_const_value(arg)
            arg_var = record!(t.tape, Constant, val)
            result[i] = arg_var
        end
    end
    return result
end


"""Push a new call frame to tracer, setting function params accordingly"""
function push_frame!(t::IRTracer, arg_defs...)
    tape_ids = ssa_args_to_tape_vars!(t, arg_defs)
    frame = Frame(Dict(i => tape_id for (i, tape_id) in enumerate(tape_ids)), -1)
    push!(t.frames, frame)
end


"""Pop call frame from tracer"""
function pop_frame!(t::IRTracer, res_sid::Int)
    frame = pop!(t.frames)
    # create mapping from the current SSA ID to the last instruction on the tape
    t.frames[end].ssa2tape[res_sid] =
        (frame.resultid == -1 ? length(t.tape) : frame.resultid)
end


"""Set target branch parameters to variables corresponding to SSA args"""
function set_branch_params!(t::IRTracer, ssa_args, target_params)
    tape_vars = ssa_args_to_tape_vars!(t, ssa_args)
    ssa2tape = t.frames[end].ssa2tape
    for (v, p) in zip(tape_vars, target_params)
        ssa2tape[p] = v
    end
end


"""Set return variable for the current frame"""
function set_return!(t::IRTracer, arg_sid_ref)
    # global STATE = (t, arg_sid_ref)
    tape_var = ssa_args_to_tape_vars!(t, [arg_sid_ref[]])[1]
    t.frames[end].resultid = tape_var
end


################################################################################
#                                  Loop utils                                  #
################################################################################


# TODO: check on several examples
function is_loop(block::IRTools.Block)
    for br in IRTools.branches(block)
        # if a branch refers to an earlier block and is not return
        # then it must be a loop
        if br.block <= block.id && br.block != 0
            return true
        end
    end
    return false
end


"""Get SSA IDs of a block's inputs"""
function block_input_ssa_ids(block::IRTools.Block)
    result = []
    for (stmt_id, (block_id, arg_id)) in enumerate(block.ir.defs)
        if block_id == block.id && arg_id < 0
            push!(result, stmt_id)
        end
    end
    return result
end


"""
Get SSA IDs of arguments which are not part of this block
(e.g. come from outside of a loop)
"""
function block_outsider_ssa_ids(block::IRTools.Block)
    result = []
    min_id = minimum(block_input_ssa_ids(block))
    for (v, stmt) in block
        ex = stmt.expr
        @assert Meta.isexpr(ex, :call)
        for arg in ex.args[2:end]
            if arg isa IRTools.Variable && arg.id < min_id
                push!(result, arg.id)
            end
        end
    end
    return result
end


function loop_exit_branch(block::IRTools.Block)
    branches = IRTools.branches(block)
    return branches[findfirst([br.block > block.id for br in branches])]
end


function loop_condition_ssa_id(block::IRTools.Block)
    br = loop_exit_branch(block)
    return br.condition.id
end


"""Get SSA IDs of exit arguments of a loop block"""
function loop_exit_ssa_ids(block::IRTools.Block)
    br = loop_exit_branch(block)
    return br.args
end


"""Pseudo op to designate loop end. Removed after Loop op is created"""
mutable struct _LoopEnd <: AbstractOp
    id::Int
end


"""
Trigger loop start operations.

Arguments:
----------

 * t :: IRTracer
    Current tracer
 * loop_input_ssa_ids :: Vector{Int}
    SSA IDs of variables which will be used as loop inputs. Includes
    loop block inputs and any outside IDs
"""
function enter_loop!(t::IRTracer, loop_input_ssa_ids::Vector)
    # skip if it's not the first iteration
    t.tape.traced && return
    # create subtape, with the current tape as parent
    subtape = Tape()
    subtape.parent = t.tape
    # create a new frame and push to the list
    frame = Frame(Dict(), -1)
    push!(t.frames, frame)
    # record inputs to the subtape & populate the new frame's ssa2id
    for ssa_id in loop_input_ssa_ids
        parent_tape_id = t.frames[end - 1].ssa2tape[ssa_id]
        val = subtape.parent[parent_tape_id].val
        tape_id = record!(subtape, Input, val)
        t.frames[end].ssa2tape[ssa_id] = tape_id
    end
    # replace IRTracer.tape with subtape
    t.tape = subtape
end


"""
Trigget loop end operations
"""
function exit_loop!(t::IRTracer,
                    input_ssa_ids::Vector,
                    cond_ssa_id::Any,
                    exit_ssa_ids::Vector)
    # set flag to stop creatnig new subtapes
    t.tape.traced = true
    # record a special op to designate the end of the loop code
    # tracer will continue to record ops, but later we truncate
    # the tape to get only ops before _LoopEnd
    record!(t.tape, _LoopEnd)
    # loop subtape already contains a variable designating condition
    # of loop continuation; if this condition is false,
    # we are ready to exit the loop and record Loop operation
    cond_var = t.tape[t.frames[end].ssa2tape[cond_ssa_id]]
    if !cond_var.val
        # swap active tape back
        loop_frame = pop!(t.frames)
        ssa2tape = loop_frame.ssa2tape
        parent_ssa2tape = t.frames[end].ssa2tape
        subtape = t.tape
        t.tape = t.tape.parent
        # record loop operation        
        parent_input_ids = [parent_ssa2tape[ssa_id] for ssa_id in input_ssa_ids]
        cond_id = ssa2tape[cond_ssa_id]
        exit_id = -1   # TODO: fill these
        loop_id = record!(t.tape, Loop, parent_input_ids, cond_id, exit_id, subtape)
    end
    # TODO:
    # 2. Record a tuple of output branch targets as return value
    # 3. Create a loop operator on the current tape,
    # 4. Optimize the loop / record conditions / finish the loop logic
    # 5. t.tape = t.tape.parent

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


function record_or_recurse!(t::IRTracer, res_sid::Int, farg_defs, fargs...)
    fn, args = fargs[1], fargs[2:end]
    # global STATE = (t, res_sid, farg_defs, fargs)
    if fn in t.primitives || (fn isa Type && fn <: NamedTuple)
        res = fn(args...)
        tape_ids = ssa_args_to_tape_vars!(t, farg_defs[2:end])
        # record corresponding op to the tape
        res_tid = record!(t.tape, Call, res, fn, tape_ids)
        # update mapping from SSA var to tape var
        # note that in functions with loops this mapping may change over time
        t.frames[end].ssa2tape[res_sid] = res_tid
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
    t.frames[end].ssa2tape[res_sid] = res_tid
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


function trace_loops!(ir::IR)
    for block in IRTools.blocks(ir)
        if is_loop(block)
            # loop block start
            loop_input_ssa_ids = vcat(
                block_input_ssa_ids(block),
                block_outsider_ssa_ids(block),
            )
            pushfirst!(block, Expr(:call, enter_loop!, self, loop_input_ssa_ids))
            # loop block end
            # TODO: find all variables with id less than the first input
            # pass them as additional inputs (ssa2tape[arg_id] = new_tape_id)
            # When creating LoopOp pass these additional inputs as well
            push!(block, Expr(:call, exit_loop!, self,
                              loop_input_ssa_ids,
                              loop_condition_ssa_id(block),
                              loop_exit_ssa_ids(block)))
        end
    end
end



@dynamo function (t::IRTracer)(fargs...)
    ir = IR(fargs...)
    ir === nothing && return   # intrinsic functions
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
    trace_loops!(ir)
    return ir
end


# @dynamo function (t::IRTracer)(fargs...)
#     ir = IR(fargs...)
#     ir === nothing && return   # intrinsic functions
#     rewrite_special_cases!(ir)
#     for block in IRTools.blocks(ir)
#         ex = st.expr
#         # note the difference:
#         # * `ex.args` is an array and thus will be passed to a function as is,
#         # including definitions of IRTools.Variable;
#         # * `ex.args...` is top-level to this expression and thus all Variable's
#         # will be replaced with actual values during runtime
#         if Meta.isexpr(ex, :call)
#             ir[v] = IRTools.xcall(record_or_recurse!, self, v.id, ex.args, ex.args...)
#         elseif is_loop(block)
#         else
#             # e.g. GlobalRef
#             # note: using insertafter!() due to
#             # https://github.com/FluxML/IRTools.jl/issues/78
#             # ir[v] = Expr(:call, record_const!, self, v.id, v)
#             insertafter!(ir, v, IRTools.xcall(record_const!, self, v.id, v))
#         end
#     end
#     trace_branches!(ir)
#     return ir
# end


"""
Trace function execution, returning its value and the generated tape.
"""
function trace(f, args...; primitives=PRIMITIVES, optimize=true)
    t = IRTracer(; primitives=primitives)
    for arg in args
        record!(t.tape, Input, arg)
    end
    push!(t.frames, Frame(Dict(i + 1 => i for i in 1:length(args)), -1))
    val = t(f, args...)
    t.tape.resultid = t.frames[1].resultid
    tape = t.tape
    if optimize
        tape = simplify(tape)
    end
    return val, tape
end
