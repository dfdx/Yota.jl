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
#                        IRTracer (defs and utils)                             #
################################################################################

mutable struct IRTracer
    is_primitive::Function
    tape::Tape
    frames::Vector{Frame}
end

function IRTracer(; ctx=Dict(), is_primitive=is_chainrules_primitive)
    tape = Tape(ctx)
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


function block_input_ssa_ids(block::IRTools.Block)
    result = []
    for (stmt_id, (block_id, arg_id)) in enumerate(block.ir.defs)
        if block_id == block.id && arg_id < 0
            push!(result, stmt_id)
        end
    end
    return result
end


<<<<<<< HEAD
function enter_loop!(t::IRTracer, input_ssa_ids::Vector)  # Int or IRTools.Variable?
=======
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
>>>>>>> e44bdb1 (Add loop outisder variables as its inputs)

    # TODO:
    # 1.
    # 2.
    # 3.
    # 4. replace IRTracer.tape with subtape
    # 5. Set .traced = false (?)

<<<<<<< HEAD
=======
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
>>>>>>> 8631569 (Partially implement exit_loop!() in the new tracer)
    # create subtape, with the current tape as parent
    subtape = Tape()
    subtape.parent = t.tape
<<<<<<< HEAD
    t.tape = subtape
    # create and push a new frame
    # frame = Frame(Dict(), -1)
    # push!(t.frames, frame)
    # record inputs to the subtape & populate the new frame's ssa2id

    # TODO: current error happens because loop IR refers to SSA vars from outside
    # the loop. We should either use the same frame, or copy parent
    # frame's ssa2tape to the child's
    
    for ssa_id in input_ssa_ids
        parent_tape_id = t.frames[end].ssa2tape[ssa_id]
        val = t.tape.parent[parent_tape_id].val
        tape_id = record!(t.tape, Input, val)
=======
    # create a new frame and push to the list
    frame = Frame(Dict(), -1)
    push!(t.frames, frame)
    # record inputs to the subtape & populate the new frame's ssa2id
    for ssa_id in loop_input_ssa_ids
        parent_tape_id = t.frames[end - 1].ssa2tape[ssa_id]
        val = subtape.parent[parent_tape_id].val
        tape_id = record!(subtape, Input, val)
>>>>>>> e44bdb1 (Add loop outisder variables as its inputs)
        t.frames[end].ssa2tape[ssa_id] = tape_id
    end
<<<<<<< HEAD


end


function exit_loop!(t::IRTracer)
    # TODO:
    # 1. Set .traced = true (?)
=======
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
<<<<<<< HEAD
<<<<<<< HEAD
    # TODO:   
>>>>>>> 8631569 (Partially implement exit_loop!() in the new tracer)
=======
=======
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
>>>>>>> a745a45 (Create (incomplete) Loop op on parent tape)
    # TODO:
>>>>>>> e44bdb1 (Add loop outisder variables as its inputs)
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


function trace_loops!(ir::IR)
    for block in IRTools.blocks(ir)
        if is_loop(block)
<<<<<<< HEAD
<<<<<<< HEAD
            pushfirst!(block, Expr(:call, enter_loop!, self, block_input_ssa_ids(block)))
<<<<<<< HEAD
            push!(block, Expr(:call, exit_loop!, self))
=======
=======
            # loop block start            
=======
            # loop block start
>>>>>>> a745a45 (Create (incomplete) Loop op on parent tape)
            loop_input_ssa_ids = vcat(
                block_input_ssa_ids(block),
                block_outsider_ssa_ids(block),
            )
            pushfirst!(block, Expr(:call, enter_loop!, self, loop_input_ssa_ids))
>>>>>>> e44bdb1 (Add loop outisder variables as its inputs)
            # loop block end
            # TODO: find all variables with id less than the first input
            # pass them as additional inputs (ssa2tape[arg_id] = new_tape_id)
            # When creating LoopOp pass these additional inputs as well
<<<<<<< HEAD
            push!(block, Expr(:call, exit_loop!, self, loop_exit_ssa_ids(block)))
>>>>>>> 8631569 (Partially implement exit_loop!() in the new tracer)
=======
            push!(block, Expr(:call, exit_loop!, self,
                              loop_input_ssa_ids,
                              loop_condition_ssa_id(block),
                              loop_exit_ssa_ids(block)))
>>>>>>> a745a45 (Create (incomplete) Loop op on parent tape)
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
