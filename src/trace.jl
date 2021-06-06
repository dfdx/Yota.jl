import IRTools
import IRTools: IR, @dynamo, self, insertafter!
import UUIDs: UUID, uuid1


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
     __new__, namedtuple]);


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
#                              Tracer Options                                  #
################################################################################

const TRACING_OPTIONS = Ref(Dict())

"""
    should_trace_loops!(val=false)

Turn on/off loop tracing. Without parameters, resets the flag to the default value
"""
should_trace_loops!(val::Bool=false) = (TRACING_OPTIONS[][:trace_loops] = val)
should_trace_loops() = get(TRACING_OPTIONS[], :trace_loops, false)


"""
Tracer options. Configured globally via the following methods:

- should_trace_loops!()
"""
struct TracerOptions
    trace_loops::Bool
end


TracerOptions() = TracerOptions(should_trace_loops())


################################################################################
#                        IRTracer (defs and utils)                             #
################################################################################

mutable struct IRTracer
    is_primitive::Function
    tape::Tape
    frames::Vector{Frame}
    options::TracerOptions
end

function IRTracer(; ctx=Dict(), is_primitive=is_chainrules_primitive)
    tape = Tape(ctx)
    return IRTracer(is_primitive, tape, [], TracerOptions())
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
    @assert(arg_sid_ref[] !== nothing,
    "Cannot set return value to nothing. Does this function actually return a value?")
    tape_var = get_tape_vars(t, [arg_sid_ref[]])[1]
    t.frames[end].result = tape_var
end


################################################################################
#                                  Loop utils                                  #
################################################################################


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
function block_input_ir_ids(block::IRTools.Block)
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
function block_outsider_ir_ids(block::IRTools.Block)
    result = []
    min_id = minimum(block_input_ir_ids(block))
    for (_, stmt) in block
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
    target_branch_id = findfirst([br.block > block.id for br in branches])
    return target_branch_id !== nothing ? branches[target_branch_id] : nothing
end


function loop_continue_branch(block::IRTools.Block)
    branches = IRTools.branches(block)
    target_branch_id = findfirst([br.block <= block.id for br in branches])
    return return target_branch_id !== nothing ? branches[target_branch_id] : nothing
end


function loop_condition_ir_id(block::IRTools.Block)
    br = loop_exit_branch(block)
    return br !== nothing ? br.condition.id : nothing
end


"""Get SSA IDs of exit arguments of a loop block"""
function loop_exit_ir_ids(block::IRTools.Block)
    br = loop_exit_branch(block)
    return br !== nothing ? [arg.id for arg in br.args] : nothing
end


function loop_continue_ir_ids(block::IRTools.Block)
    br = loop_continue_branch(block)
    return br !== nothing ? [arg.id for arg in br.args] : nothing
end


"""Pseudo op to designate loop end. Removed after Loop op is created"""
mutable struct _LoopEnd <: AbstractOp
    id::Int
end
_LoopEnd() = _LoopEnd(0)


const LOOP_EXIT_TAPE_IDS = "loop_exit_tape_ids"
const LOOP_COND_ID = "loop_cond_id"
const LOOP_CONTINUE_TAPE_IDS = "loop_continue_tape_ids"


is_loop_traced(t::IRTracer, loop_id::UUID) = haskey(t.tape.meta, "loop_already_traced_$loop_id")
loop_traced!(t::IRTracer, loop_id::UUID) = (t.tape.meta["loop_already_traced_$loop_id"] = true)


"""
Trigger loop start operations.

Arguments:
----------

 * t :: IRTracer
    Current tracer
 * loop_id :: Int
    Unique ID of a loop being entered
 * loop_input_ir_ids :: Vector{Int}
    IR IDs of variables which will be used as loop inputs. Includes
    loop block inputs and any outside IDs

This function is added to the very beginning of the loop block(s).
During the first iteration we initialize a subtape which will be used
later to create the Loop operation on the parent tape. Since all iterations
of the loop produce identical operations, we only need to trace it once.
However, it turns out to be easier to record all iterations (seprated by
a special _LoopEnd op) and then prune unused iterations.

Another important detail is that Loop's subtape is a completely valid
and independent tape with its own call frame and inputs which include
all explicit and implicit inputs to the loop's block in the original IR.
"""
function enter_loop!(t::IRTracer, loop_id, loop_input_ir_ids::Vector)
    t.options.trace_loops || return
    # skip if it's not the first iteration
    is_loop_traced(t, loop_id) && return
    # create subtape, with the current tape as parent
    C = typeof(t.tape.c)
    subtape = Tape(C())
    subtape.parent = t.tape
    # create a new frame and push to the list
    frame = Frame(Dict(), V(0), ())
    push!(t.frames, frame)
    # record inputs to the subtape & populate the new frame's ir2tape
    for ir_id in loop_input_ir_ids
        parent_tape_var = t.frames[end - 1].ir2tape[ir_id]
        val = subtape.parent[parent_tape_var].val
        tape_var = push!(subtape, Input(val))
        t.frames[end].ir2tape[ir_id] = tape_var
    end
    # replace IRTracer.tape with subtape
    t.tape = subtape
end


"""
Set flags designating the end of the first run of the loop.
"""
function stop_loop_tracing!(t::IRTracer,
                            loop_id::Any,
                            input_ir_ids::Vector,
                            cond_ir_id::Any,
                            cont_ir_ids::Vector,
                            exit_ir_ids::Vector,
                            exit_target_ir_ids::Vector)
    t.options.trace_loops || return
    if !is_loop_traced(t, loop_id)
        # record exit tape IDs as of first iteration
        # we will use them later
        t.tape.meta[LOOP_EXIT_TAPE_IDS] =
            [t.frames[end].ir2tape[ir_id] for ir_id in exit_ir_ids]
        t.tape.meta[LOOP_COND_ID] = t.frames[end].ir2tape[cond_ir_id]
        t.tape.meta[LOOP_CONTINUE_TAPE_IDS] =
            [t.frames[end].ir2tape[ir_id] for ir_id in cont_ir_ids]
        # set flag to stop creatnig new subtapes
        loop_traced!(t, loop_id)
    end
    # record a special op to designate the end of the loop code
    # tracer will continue to record ops, but later we truncate
    # the tape to get only ops before _LoopEnd
    push!(t.tape, _LoopEnd())
end


"""
Trigget loop end operations.

This function is added just before the end of the loop block.
Since we record all iterations of the loop, we must remember tape IDs
of continuation condition and exit variables during the first run.
"""
function exit_loop!(t::IRTracer,
                    input_ir_ids::Vector,
                    cond_ir_id::Any,
                    cont_ir_ids::Vector,
                    exit_ir_ids::Vector,
                    exit_target_ir_ids::Vector)
    t.options.trace_loops || return
    # loop subtape already contains a variable designating condition
    # of loop continuation; if this condition is false,
    # we are ready to exit the loop and record Loop operation
    cond_op = t.tape[t.frames[end].ir2tape[cond_ir_id]]
    if !cond_op.val
        # swap active tape back
        pop!(t.frames)
        parent_ir2tape = t.frames[end].ir2tape
        subtape = t.tape
        t.tape = t.tape.parent
        # remove repeating blocks
        first_loop_end = findfirst(op -> isa(op, _LoopEnd), subtape.ops)
        subtape.ops = subtape.ops[1:first_loop_end-1]
        # record output tuple
        exit_tape_vars = subtape.meta[LOOP_EXIT_TAPE_IDS]
        # exit_val = tuple([subtape[id].val for id in exit_tape_vars]...)
        # exit_var = push!(subtape, Call, exit_val, tuple, exit_tape_ids)
        exit_var = push!(subtape, mkcall(tuple, exit_tape_vars...))
        exit_val = subtape[exit_var].val
        subtape.result = exit_var
        # record the loop operation
        # global STATE = (t, input_ir_ids, exit_target_ir_ids)
        parent_input_vars = [parent_ir2tape[ir_id] for ir_id in input_ir_ids]
        condition = subtape.meta[LOOP_COND_ID]
        cont_vars = subtape.meta[LOOP_CONTINUE_TAPE_IDS]
        loop_id = push!(t.tape, Loop(0, parent_input_vars,
                          condition, cont_vars,
                          exit_var, subtape, exit_val))
        # destructure loop return values to separate vars on the main tape
        # and map branch arguments to these vars
        for i=1:length(exit_val)
            res_id = push!(t.tape, mkcall(getfield, loop_id, i))
            parent_ir2tape[exit_target_ir_ids[i]] = res_id
        end
    end
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


"""
Get SSA IDs of the branch target parameters.
For example, given code like this:


  2: (%9, %10, %11)
    ...
    br 3 (%14, %15) unless %18
    br 2 (%16, %14, %15)
  3: (%19, %20)
    ...

This function will return:

  branch_target_params(ir, <br 3>) ==> [19, 20]

"""
function branch_target_params(ir:: IR, branch::IRTools.Branch)
    return [v.id for v in ir.blocks[branch.block].args]
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


function loop_start_end_blocks(ir::IR, block::IRTools.Block)
    start_block = IRTools.block(ir, loop_continue_branch(block).block)
    exit_branch = loop_exit_branch(block)
    # check if this block contains the exit branch
    if exit_branch !== nothing && exit_branch.block > block.id
        end_block = block
    else
        exit_branch = loop_exit_branch(start_block)
        if exit_branch !== nothing && exit_branch.block > block.id
            end_block = start_block
        else
            error("Cannot find end block of a loop")
        end
    end
    return start_block, end_block
end


function trace_loops!(ir::IR)
    for block in IRTools.blocks(ir)
        if is_loop(block)
            loop_id = uuid1()   # unique ID of this loop
            start_block, end_block = loop_start_end_blocks(ir, block)
            # loop start - the first block of the loop
            loop_input_ir_ids = vcat(
                block_input_ir_ids(start_block),
                block_outsider_ir_ids(start_block),
            )
            pushfirst!(start_block, Expr(:call, enter_loop!, self, loop_id, loop_input_ir_ids))
            # loop tracing border - at this point all operations of the loop
            # have been executed at least once, even if continuation condition
            # is in another block
            push!(block, Expr(:call, stop_loop_tracing!, self,
                              loop_id,
                              loop_input_ir_ids,
                              loop_condition_ir_id(end_block),
                              loop_continue_ir_ids(block),
                              loop_exit_ir_ids(end_block),
                              branch_target_params(ir, loop_exit_branch(end_block))))
            # loop end - continuation condition is checked here
            push!(end_block, Expr(:call, exit_loop!, self,
                              loop_input_ir_ids,
                              loop_condition_ir_id(end_block),
                              loop_continue_ir_ids(block),
                              loop_exit_ir_ids(end_block),
                              branch_target_params(ir, loop_exit_branch(end_block))))
        end
    end
end


@dynamo function (t::IRTracer)(fargs...)
    ir = IR(fargs...)
    ir === nothing && return   # intrinsic functions
    IRTools.expand!(ir)
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
function trace(f, args...; is_primitive=is_primitive, primitives=nothing, ctx=Dict())
    if primitives !== nothing
        sigs = FunctionResolver{Bool}([Tuple{typeof(f), Vararg} => true for f in primitives])
        is_primitive = sig -> sig in sigs
    end
    t = IRTracer(; ctx=ctx, is_primitive=is_primitive)
    arg_vars = inputs!(t.tape, f, args...)
    frame = Frame(
        Dict(i => a for (i, a) in enumerate(arg_vars)), V(0), (f, args...)
    )
    push!(t.frames, frame)
    val = t(f, args...)
    t.tape.result = t.frames[1].result
    tape = t.tape
    return val, tape
end
